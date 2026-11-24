"""Train the 2D latent PDE operator or a parameter-matched FNO baseline."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from baselines_2d import DeepONet2D, DeepONet2DConfig, FNO2D, FNO2DConfig
from coord_attention_operator_2d import (
    CoordAttentionOperator2D,
    Operator2DConfig,
)
from operator_2d_common import (
    configure_backend,
    count_parameters,
    divergence_relative_loss,
    load_npz_split,
    load_spec,
    pde_residual_2d,
    relative_l2,
    set_seed,
    space_time_coordinates,
)


def build_parser(model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("pg2d", "fno2d", "deeponet2d"),
        default=model,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--dev-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--test-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--initial-weight", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode",
        choices=("mse", "relative"),
        default="relative",
    )
    parser.add_argument("--first-frame-weight", type=float, default=0.05)
    parser.add_argument("--recon-weight", type=float, default=0.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument("--divergence-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    fields = (
        (
            Operator2DConfig.__dataclass_fields__
            if model == "pg2d"
            else (
                FNO2DConfig.__dataclass_fields__
                if model == "fno2d"
                else DeepONet2DConfig.__dataclass_fields__
            )
        )
    )
    for name, field in fields.items():
        option = "--" + name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(
                option,
                action=argparse.BooleanOptionalAction,
                default=field.default,
            )
        else:
            parser.add_argument(
                option,
                type=type(field.default),
                default=field.default,
            )
    return parser


@torch.no_grad()
def evaluate(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 2,
) -> float:
    model.eval()
    prediction = []
    for start in range(0, inputs.size(0), batch_size):
        stop = min(start + batch_size, inputs.size(0))
        prediction.append(
            model.forward_grid(inputs[start:stop].to(device)).cpu()
        )
    model.train()
    return relative_l2(torch.cat(prediction), targets)


@torch.no_grad()
def evaluate_divergence(
    model,
    inputs: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 2,
) -> float:
    model.eval()
    values = []
    for start in range(0, inputs.size(0), batch_size):
        stop = min(start + batch_size, inputs.size(0))
        prediction = model.forward_grid(inputs[start:stop].to(device))
        values.append(
            float(divergence_relative_loss(prediction, coordinates).detach())
        )
    model.train()
    return float(sum(values) / max(len(values), 1))


def save_checkpoint(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    model_name: str,
    config: dict,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "model_name": model_name,
            "config": config,
            **extra,
        },
        path,
    )


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--model",
        choices=("pg2d", "fno2d", "deeponet2d"),
        default="pg2d",
    )
    known, _ = pre.parse_known_args()
    args = build_parser(known.model).parse_args()
    if args.model == "pg2d":
        config_cls = Operator2DConfig
    elif args.model == "fno2d":
        config_cls = FNO2DConfig
    else:
        config_cls = DeepONet2DConfig
    if args.quick:
        args.steps = 30
        args.eval_every = 10
        args.batch_size = 2
        args.train_samples = min(args.train_samples or 8, 8)
        args.dev_samples = min(args.dev_samples or 8, 8)
        args.test_samples = min(args.test_samples or 8, 8)
        args.early_stop_patience = 3
        args.warmup = min(args.warmup, 5)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(device)
    spec = load_spec(args.data_dir / "metadata.json")
    train_inputs, train_targets = load_npz_split(
        args.data_dir / "train.npz"
        if (args.data_dir / "train.npz").exists()
        else args.data_dir / f"{args.data_dir.name}_train.npz"
    )
    dev_inputs, dev_targets = load_npz_split(
        args.data_dir / "dev.npz"
        if (args.data_dir / "dev.npz").exists()
        else args.data_dir / f"{args.data_dir.name}_dev.npz"
    )
    test_inputs, test_targets = load_npz_split(
        args.data_dir / "test.npz"
        if (args.data_dir / "test.npz").exists()
        else args.data_dir / f"{args.data_dir.name}_test.npz"
    )
    expected_input = (spec.effective_input_channels, spec.nx, spec.ny)
    expected_target = (spec.channels, spec.nx, spec.ny, spec.nt)
    for split, inputs, targets in (
        ("train", train_inputs, train_targets),
        ("dev", dev_inputs, dev_targets),
        ("test", test_inputs, test_targets),
    ):
        if tuple(inputs.shape[1:]) != expected_input:
            raise ValueError(
                f"{split} input shape {tuple(inputs.shape)} "
                f"does not match {expected_input}"
            )
        if tuple(targets.shape[1:]) != expected_target:
            raise ValueError(
                f"{split} target shape {tuple(targets.shape)} "
                f"does not match {expected_target}"
            )
    if args.train_samples > 0:
        train_inputs = train_inputs[: args.train_samples]
        train_targets = train_targets[: args.train_samples]
    if args.dev_samples > 0:
        dev_inputs = dev_inputs[: args.dev_samples]
        dev_targets = dev_targets[: args.dev_samples]
    if args.test_samples > 0:
        test_inputs = test_inputs[: args.test_samples]
        test_targets = test_targets[: args.test_samples]
    print(
        f"train {tuple(train_inputs.shape)} "
        f"dev {tuple(dev_inputs.shape)} "
        f"test {tuple(test_inputs.shape)}",
        flush=True,
    )

    checkpoint = None
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    if args.evaluate_only:
        if checkpoint is None:
            raise ValueError("--evaluate-only requires --init-checkpoint")
        cfg = config_cls(**checkpoint["config"])
        cfg.field_channels = spec.channels
        cfg.input_channels = spec.effective_input_channels
    else:
        cfg = config_cls(
            **{
                name: getattr(args, name)
                for name in config_cls.__dataclass_fields__
            }
        )
        cfg.field_channels = spec.channels
        cfg.input_channels = spec.effective_input_channels

    if args.model == "pg2d":
        model = CoordAttentionOperator2D(
            cfg,
            spec.nx,
            spec.ny,
            spec.nt,
        ).to(device)
    elif args.model == "fno2d":
        model = FNO2D(cfg, spec.nx, spec.ny, spec.nt).to(device)
    else:
        model = DeepONet2D(
            cfg,
            spec.nx,
            spec.ny,
            spec.nt,
            spec.x_min,
            spec.x_max,
            spec.y_min,
            spec.y_max,
            spec.t_min,
            spec.t_max,
        ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"initialized from {args.init_checkpoint}", flush=True)

    parameter_count = count_parameters(model)
    print(f"model {args.model} params {parameter_count}", flush=True)
    if args.evaluate_only:
        result = {
            "dev_relative_l2": evaluate(
                model,
                dev_inputs,
                dev_targets,
                device,
                args.batch_size,
            ),
            "test_relative_l2": evaluate(
                model,
                test_inputs,
                test_targets,
                device,
                args.batch_size,
            ),
        }
        print(json.dumps(result, indent=2), flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "evaluation_only": True,
                    "config": asdict(cfg),
                    "parameter_count": parameter_count,
                    "grid": asdict(spec),
                    **result,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    if args.recon_weight > 0.0 and args.model != "pg2d":
        raise ValueError("reconstruction loss is only available for pg2d")
    if args.recon_weight > 0.0 and not cfg.latent_recon:
        raise ValueError("--recon-weight requires --latent-recon")
    if args.residual_weight > 0.0 and (
        args.model != "pg2d" or spec.channels != 2
    ):
        raise ValueError(
            "Gray-Scott residual loss requires pg2d with two channels"
        )
    if args.divergence_weight > 0.0 and spec.channels < 2:
        raise ValueError(
            "incompressible divergence loss requires at least u/v channels"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        progress = min(
            1.0,
            (step - args.warmup)
            / max(1, args.steps - args.warmup),
        )
        return args.min_lr_ratio + (
            1.0 - args.min_lr_ratio
        ) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    coordinates = space_time_coordinates(spec, device)
    history = []
    best_dev = float("inf")
    best_step = -1
    best_state = None
    stale = 0
    stopped_early = False
    started = time.time()

    for step in range(args.steps):
        indices = torch.randint(
            0,
            train_inputs.size(0),
            (args.batch_size,),
        )
        inputs = train_inputs[indices].to(device)
        targets = train_targets[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            enabled=args.amp and device.type == "cuda",
        ):
            prediction = model.forward_grid(inputs)
            difference = (prediction - targets).square()
            if args.loss_mode == "mse":
                data_loss = torch.mean(difference)
            else:
                target_center = targets.mean(
                    dim=(2, 3),
                    keepdim=True,
                )
                target_variance = (
                    (targets - target_center)
                    .square()
                    .mean(dim=(2, 3))
                    .clamp_min(1.0e-4)
                )
                per_time = (
                    difference.mean(dim=(2, 3))
                    / target_variance
                ).mean(dim=0)
                data_loss = per_time[:, 1:].mean()
                data_loss = (
                    data_loss
                    + args.first_frame_weight * per_time[:, 0].mean()
                )
            loss = data_loss
            if args.initial_weight > 0.0:
                loss = loss + args.initial_weight * torch.mean(
                    (prediction[..., 0] - targets[..., 0]).square()
                )
            if args.recon_weight > 0.0:
                reconstruction = model.reconstruct(inputs)
                loss = loss + args.recon_weight * torch.mean(
                    (reconstruction - inputs).square()
                )
            if args.residual_weight > 0.0:
                residual = pde_residual_2d(prediction, coordinates)
                loss = loss + args.residual_weight * torch.mean(
                    residual.square()
                )
            if args.divergence_weight > 0.0:
                loss = loss + args.divergence_weight * divergence_relative_loss(
                    prediction,
                    coordinates,
                )
        if not torch.isfinite(loss):
            print(f"step {step + 1}: skipped non-finite loss", flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            dev_l2 = evaluate(
                model,
                dev_inputs,
                dev_targets,
                device,
                args.batch_size,
            )
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "dev_l2": dev_l2,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if dev_l2 < best_dev:
                best_dev = dev_l2
                best_step = step + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            history.append(record)
            print(
                f"step {step + 1:5d} loss {record['loss']:.3e} "
                f"dev_l2 {dev_l2:.6f} best {best_dev:.6f}@{best_step} "
                f"stale {stale}",
                flush=True,
            )
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                milestone = (
                    args.output_dir
                    / "milestones"
                    / f"step_{step + 1:06d}"
                )
                save_checkpoint(
                    milestone / "model.pt",
                    model.state_dict(),
                    args.model,
                    asdict(cfg),
                    {
                        "step": step + 1,
                        "dev_l2": dev_l2,
                        "parameter_count": parameter_count,
                    },
                )
            if (
                args.early_stop_patience > 0
                and stale >= args.early_stop_patience
            ):
                stopped_early = True
                print(
                    f"early stop at step {step + 1}: "
                    f"dev has not improved for {stale} evaluations",
                    flush=True,
                )
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    result = {
        "train_relative_l2": evaluate(
            model,
            train_inputs[: min(32, train_inputs.size(0))],
            train_targets[: min(32, train_targets.size(0))],
            device,
            args.batch_size,
        ),
        "dev_relative_l2": best_dev,
        "test_relative_l2": evaluate(
            model,
            test_inputs,
            test_targets,
            device,
            args.batch_size,
        ),
    }
    if args.divergence_weight > 0.0:
        result["test_divergence_relative"] = evaluate_divergence(
            model,
            test_inputs,
            coordinates,
            device,
            args.batch_size,
        )
    print(
        f"best@{best_step} train {result['train_relative_l2']:.6f} "
        f"dev {result['dev_relative_l2']:.6f} "
        f"test {result['test_relative_l2']:.6f}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": args.model,
        "config": asdict(cfg),
        "grid": asdict(spec),
        "parameter_count": parameter_count,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "training_seconds": time.time() - started,
        "dataset_dir": str(args.data_dir),
        **result,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_checkpoint(
        args.output_dir / "model.pt",
        best_state,
        args.model,
        asdict(cfg),
        {
            "best_step": best_step,
            "parameter_count": parameter_count,
            **result,
        },
    )


if __name__ == "__main__":
    main()
