"""训练坐标查询交叉注意力算子（code/coord_attention_operator.py）。

损失只有数据 MSE：每步随机抽 batch_size 条轨迹，每条随机抽 points 个网格点。
AdamW + warmup + 余弦退火 + 梯度裁剪；每 eval_every 步在全部 dev 样本上整场评估，
按 dev 相对 L2 保留最佳权重；最后在全部 test 样本上整场评估。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from burgers_common import (
    NT,
    NX,
    configure_backend,
    count_parameters,
    grid_coordinates,
    load_split,
    relative_l2,
    resolve_split_path,
    set_seed,
)
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from pde_residual_1d import normalized_pde_residual

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "pdebench_burgers"


@torch.no_grad()
def predict_full(model, fields: torch.Tensor, device, batch: int = 8) -> torch.Tensor:
    """fields [N, nx, nt, 1] -> 整场预测 [N, nx, nt]。"""
    model.eval()
    x, t = grid_coordinates(device)
    out = []
    for s in range(0, fields.size(0), batch):
        u0 = fields[s:s + batch, :, 0, 0]
        if model.grid_decoder is not None:
            out.append(model.forward_grid(u0))
            continue
        kv, trajectory = model.encode(u0)
        b = u0.size(0)
        out.append(
            model.decode(
                kv,
                x.expand(b, -1, -1),
                t.expand(b, -1, -1),
                trajectory,
            ).view(b, NX, NT)
        )
    model.train()
    return torch.cat(out)


def evaluate(model, fields, device) -> float:
    return relative_l2(predict_full(model, fields, device), fields[..., 0])


def evaluate_with_physics(
    model,
    fields,
    device,
    equation: str,
    beta: float,
) -> dict:
    prediction = predict_full(model, fields, device)
    residual = normalized_pde_residual(prediction, equation, beta)[1]
    return {
        "l2": relative_l2(prediction, fields[..., 0]),
        "pde_relative_rms": float(residual),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-samples", type=int, default=512)
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--loss-mode", choices=("points", "full"), default="points")
    p.add_argument("--grid-chunk", type=int, default=8)
    p.add_argument("--recon-weight", type=float, default=0.0)
    p.add_argument("--physics-weight", type=float, default=0.0)
    p.add_argument(
        "--physics-equation",
        choices=("burgers", "advection"),
        default="burgers",
    )
    p.add_argument("--physics-beta", type=float, default=1.0)
    p.add_argument("--physics-start-step", type=int, default=0)
    p.add_argument("--physics-warmup", type=int, default=2000)
    p.add_argument("--physics-score-weight", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--test-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--dataset-prefix", default="burgers_nu0.01")
    p.add_argument("--dev-samples", type=int, default=10**6)
    p.add_argument("--test-samples", type=int, default=10**6)
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--quick", action="store_true")
    for name, field in OperatorConfig.__dataclass_fields__.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = p.parse_args()
    if args.quick:
        args.train_samples, args.steps, args.eval_every = 32, 60, 30
    args.grid_chunk = max(1, args.grid_chunk)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(device)
    train_path = resolve_split_path(args.data_dir, args.dataset_prefix, "train")
    dev_path = resolve_split_path(args.data_dir, args.dataset_prefix, "dev")
    test_path = resolve_split_path(args.data_dir, args.dataset_prefix, "test")
    train = load_split(train_path, args.train_samples).to(device)
    dev = load_split(dev_path, args.dev_samples).to(device)
    test = load_split(test_path, args.test_samples).to(device)
    if args.quick:
        dev, test = dev[:16], test[:16]
    print(f"train {tuple(train.shape)} dev {tuple(dev.shape)} test {tuple(test.shape)}", flush=True)

    checkpoint = None
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint,
            map_location=device,
            weights_only=False,
        )
    if args.evaluate_only and checkpoint is not None:
        known_fields = OperatorConfig.__dataclass_fields__
        operator = {
            key: value
            for key, value in checkpoint["operator"].items()
            if key in known_fields
        }
        operator["grid_transport_velocity"] = args.grid_transport_velocity
        operator["grid_transport_blend"] = args.grid_transport_blend
        operator["grid_transport_learnable"] = args.grid_transport_learnable
        operator["grid_transport_refine"] = args.grid_transport_refine
        operator["grid_transport_refine_hidden"] = args.grid_transport_refine_hidden
        cfg = OperatorConfig(**operator)
    else:
        cfg = OperatorConfig(
            **{
                key: getattr(args, key)
                for key in OperatorConfig.__dataclass_fields__
            }
        )
    model = CoordAttentionOperator(cfg).to(device)
    if checkpoint is not None:
        current = model.state_dict()
        compatible = {
            name: value
            for name, value in checkpoint["state_dict"].items()
            if name in current and current[name].shape == value.shape
        }
        x_proj_key = "grid_decoder.x_proj.weight"
        if x_proj_key in checkpoint["state_dict"] and x_proj_key in current:
            old_weight = checkpoint["state_dict"][x_proj_key]
            new_weight = current[x_proj_key]
            if (
                old_weight.ndim == new_weight.ndim == 2
                and old_weight.shape[0] == new_weight.shape[0]
                and old_weight.shape[1] % 2 == 0
                and new_weight.shape[1] % 2 == 0
            ):
                old_bands = old_weight.shape[1] // 2
                new_bands = new_weight.shape[1] // 2
                expanded = torch.zeros_like(new_weight)
                kept = min(old_bands, new_bands)
                expanded[:, :kept] = old_weight[:, :kept]
                expanded[:, new_bands : new_bands + kept] = old_weight[
                    :, old_bands : old_bands + kept
                ]
                compatible[x_proj_key] = expanded
        incompatible = model.load_state_dict(compatible, strict=False)
        buffer_names = {name for name, _ in model.named_buffers()}
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("transport") and key not in buffer_names
        ]
        if missing:
            raise ValueError(f"missing checkpoint keys: {missing}")
        print(
            f"initialized from {args.init_checkpoint} "
            f"({len(incompatible.unexpected_keys)} unused keys)",
            flush=True,
        )
    if args.evaluate_only:
        if args.init_checkpoint is None:
            raise ValueError("--evaluate-only requires --init-checkpoint")
        result = {
            "train_relative_l2": evaluate(model, train[:256], device),
            "dev_relative_l2": evaluate(model, dev, device),
            "test_relative_l2": evaluate(model, test, device),
        }
        print(
            f"evaluate-only: dev {result['dev_relative_l2']:.6f} "
            f"test {result['test_relative_l2']:.6f}",
            flush=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model": "coord_attention_latent_pde",
                    "evaluation_only": True,
                    "config": {
                        name: (
                            str(value) if isinstance(value, Path) else value
                        )
                        for name, value in vars(args).items()
                    },
                    "operator": asdict(cfg),
                    "parameter_count": count_parameters(model),
                    **result,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.save_model:
            torch.save(
                {"state_dict": model.state_dict(), "operator": asdict(cfg)},
                args.output_dir / "model.pt",
            )
        return
    if args.recon_weight > 0.0 and model.dynamics is None:
        raise ValueError("--recon-weight requires latent dynamics")
    if args.recon_weight > 0.0 and model.dynamics.encoder.decode is None:
        raise ValueError("--recon-weight requires --latent-recon true")
    if args.physics_weight > 0.0 and (
        args.loss_mode != "full" or model.grid_decoder is None
    ):
        raise ValueError(
            "--physics-weight requires --loss-mode full and grid output"
        )
    print(f"params {count_parameters(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    x_axis, t_axis = grid_coordinates(device)
    x_axis, t_axis = x_axis.view(NX, NT), t_axis.view(NX, NT)
    best, best_step, best_state, history = float("inf"), -1, None, []
    started = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, train.size(0), (args.batch_size,), device=device)
        data_loss = None
        pde_loss = None
        pde_relative = None
        physics_ramp = 0.0
        opt.zero_grad(set_to_none=True)
        if args.loss_mode == "full":
            fields = train[idx, :, :, 0]
            target = fields
            if model.grid_decoder is not None:
                prediction = model.forward_grid(fields[:, :, 0])
                data_loss = torch.mean((prediction - target) ** 2)
                loss = data_loss
                if args.recon_weight > 0.0:
                    reconstruction = model.reconstruct(fields[:, :, 0])
                    loss = loss + args.recon_weight * torch.mean(
                        (reconstruction - fields[:, :, 0]) ** 2
                    )
                pde_loss = None
                pde_relative = None
                physics_ramp = 0.0
                if args.physics_weight > 0.0:
                    pde_loss, pde_relative = normalized_pde_residual(
                        prediction,
                        args.physics_equation,
                        args.physics_beta,
                    )
                    physics_ramp = min(
                        1.0,
                        max(
                            0,
                            step + 1 - args.physics_start_step,
                        )
                        / max(args.physics_warmup, 1),
                    )
                    loss = (
                        loss
                        + args.physics_weight * physics_ramp * pde_loss
                    )
                loss.backward()
                total_points = target.numel()
                loss_value = float(loss.detach())
            else:
                context, trajectory = model.encode(fields[:, :, 0])
                total_points = target.numel()
                loss_value = 0.0
                for start in range(0, args.batch_size, args.grid_chunk):
                    stop = min(start + args.grid_chunk, args.batch_size)
                    count = stop - start
                    prediction = model.decode(
                        context[start:stop],
                        x_axis.reshape(1, -1, 1).expand(count, -1, -1),
                        t_axis.reshape(1, -1, 1).expand(count, -1, -1),
                        trajectory[start:stop] if trajectory is not None else None,
                    ).view(count, NX, NT)
                    loss_chunk = torch.sum((prediction - target[start:stop]) ** 2)
                    loss_value += float(loss_chunk.detach())
                    (loss_chunk / total_points).backward(
                        retain_graph=stop < args.batch_size
                    )
                loss = torch.tensor(loss_value / total_points, device=device)
                data_loss = loss
        else:
            xi = torch.randint(0, NX, (args.batch_size, args.points), device=device)
            ti = torch.randint(0, NT, (args.batch_size, args.points), device=device)
            fields = train[idx, :, :, 0]
            target = fields[torch.arange(args.batch_size, device=device)[:, None], xi, ti]
            pred = model(fields[:, :, 0], x_axis[xi, ti].unsqueeze(-1), t_axis[xi, ti].unsqueeze(-1))
            loss = torch.mean((pred - target) ** 2)
            data_loss = loss
            if args.recon_weight > 0.0:
                reconstruction = model.reconstruct(fields[:, :, 0])
                loss = loss + args.recon_weight * torch.mean(
                    (reconstruction - fields[:, :, 0]) ** 2
                )
        if not torch.isfinite(loss):
            print(f"step {step + 1:6d} skipped non-finite loss", flush=True)
            continue
        if args.loss_mode != "full":
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            if args.physics_weight > 0.0:
                dev_metrics = evaluate_with_physics(
                    model,
                    dev,
                    device,
                    args.physics_equation,
                    args.physics_beta,
                )
            else:
                dev_metrics = {"l2": evaluate(model, dev, device)}
            dev_l2 = dev_metrics["l2"]
            dev_pde = dev_metrics.get("pde_relative_rms")
            selection_score = dev_l2
            if dev_pde is not None:
                selection_score += args.physics_score_weight * dev_pde
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "data_loss": (
                        float(data_loss.detach())
                        if data_loss is not None
                        else None
                    ),
                    "pde_loss": (
                        float(pde_loss.detach())
                        if pde_loss is not None
                        else None
                    ),
                    "pde_relative_rms": (
                        float(pde_relative.detach())
                        if pde_relative is not None
                        else None
                    ),
                    "physics_ramp": physics_ramp,
                    "dev_l2": dev_l2,
                    "dev_pde_relative_rms": dev_pde,
                    "selection_score": selection_score,
                }
            )
            if selection_score < best:
                best = selection_score
                best_step = step + 1
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"step {step + 1:6d} loss {loss.item():.3e} "
                f"dev_l2 {dev_l2:.4f} "
                f"dev_pde {(dev_pde if dev_pde is not None else float('nan')):.4f} "
                f"best {best:.4f}@{best_step}",
                flush=True,
            )
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                milestone_dir = args.output_dir / "milestones" / f"step_{step + 1:06d}"
                milestone_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict(), "operator": asdict(cfg)},
                    milestone_dir / "model.pt",
                )
                milestone = {
                    "step": step + 1,
                    "dev_l2": dev_l2,
                    "dev_pde_relative_rms": dev_pde,
                    "parameter_count": count_parameters(model),
                }
                if args.test_every > 0 and (step + 1) % args.test_every == 0:
                    if args.physics_weight > 0.0:
                        test_metrics = evaluate_with_physics(
                            model,
                            test,
                            device,
                            args.physics_equation,
                            args.physics_beta,
                        )
                        milestone["test_l2"] = test_metrics["l2"]
                        milestone["test_pde_relative_rms"] = test_metrics[
                            "pde_relative_rms"
                        ]
                    else:
                        milestone["test_l2"] = evaluate(model, test, device)
                    print(
                        f"milestone {step + 1:6d} test_l2 "
                        f"{milestone['test_l2']:.6f}",
                        flush=True,
                    )
                (milestone_dir / "metrics.json").write_text(
                    json.dumps(milestone, indent=2),
                    encoding="utf-8",
                )

    model.load_state_dict(best_state)
    train_l2 = evaluate(model, train[:256], device)
    if args.physics_weight > 0.0:
        final_dev = evaluate_with_physics(
            model,
            dev,
            device,
            args.physics_equation,
            args.physics_beta,
        )
        final_test = evaluate_with_physics(
            model,
            test,
            device,
            args.physics_equation,
            args.physics_beta,
        )
    else:
        final_dev = {"l2": evaluate(model, dev, device)}
        final_test = {"l2": evaluate(model, test, device)}
    res = {
        "train_l2": train_l2,
        "dev_l2": final_dev["l2"],
        "test_l2": final_test["l2"],
        "dev_pde_relative_rms": final_dev.get("pde_relative_rms"),
        "test_pde_relative_rms": final_test.get("pde_relative_rms"),
    }
    print(
        f"best@{best_step}: train(256) {res['train_l2']:.4f} "
        f"dev {res['dev_l2']:.4f} "
        f"test(all {test.size(0)}) {res['test_l2']:.4f}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata_path = args.data_dir / "metadata.json"
    dataset_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "model": (
            "coord_attention_latent_pde"
            if cfg.dynamics == "latent_pde"
            else "coord_attention"
        ), "config": run_cfg, "operator": asdict(cfg),
        "dataset": {
            "prefix": args.dataset_prefix,
            "metadata": dataset_metadata,
            "time_normalization": "101 uniform snapshots mapped to model domain [0, 2]",
        },
        "parameter_count": count_parameters(model), "best_step": best_step,
        "train_relative_l2": res["train_l2"], "dev_relative_l2": res["dev_l2"],
        "test_relative_l2": res["test_l2"],
        "dev_pde_relative_rms": res["dev_pde_relative_rms"],
        "test_pde_relative_rms": res["test_pde_relative_rms"],
        "test_samples": test.size(0),
        "training_seconds": time.time() - started, "history": history}, indent=2), encoding="utf-8")
    if args.save_model:
        torch.save({"state_dict": best_state, "operator": asdict(cfg)}, args.output_dir / "model.pt")


if __name__ == "__main__":
    main()
