"""Fine-tune a PG checkpoint with a normalized PDE residual."""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from burgers_common import count_parameters, load_split, relative_l2, set_seed
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from pde_residual_1d import normalized_pde_residual


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "pdebench_burgers_official"
DEFAULT_SOURCE = (
    ROOT
    / "experiments"
    / "outputs_burgers_dataset"
    / "official_n8000"
    / "latent_pde_all_lowrank_joint"
    / "model.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "outputs_burgers_dataset"
    / "official_n8000"
    / "latent_pde_physics_tuned"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset-prefix", default="burgers_nu0.01")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--physics-weight", type=float, default=5.0e-4)
    parser.add_argument("--physics-warmup", type=int, default=500)
    parser.add_argument("--score-weight", type=float, default=0.01)
    parser.add_argument("--max-l2-increase", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--equation", choices=("burgers", "advection"), default="burgers")
    parser.add_argument("--pde-beta", type=float, default=1.0)
    parser.add_argument("--physics-refiner", action="store_true")
    parser.add_argument("--refiner-hidden", type=int, default=16)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def predict(
    model: CoordAttentionOperator,
    fields: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, fields.size(0), batch_size):
            predictions.append(
                model.forward_grid(
                    fields[start : start + batch_size, :, 0, 0]
                )
            )
    model.train()
    return torch.cat(predictions, dim=0)


def evaluate(
    model: CoordAttentionOperator,
    fields: torch.Tensor,
    device: torch.device,
    equation: str,
    beta: float,
) -> dict:
    prediction = predict(model, fields, device)
    residual = normalized_pde_residual(prediction, equation, beta)[1]
    return {
        "l2": relative_l2(prediction, fields[..., 0]),
        "pde_relative_rms": float(residual),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.source_checkpoint,
        map_location=device,
        weights_only=False,
    )
    operator = dict(checkpoint["operator"])
    operator["grid_physics_refiner"] = args.physics_refiner
    operator["grid_physics_refiner_hidden"] = args.refiner_hidden
    cfg = OperatorConfig(**operator)
    model = CoordAttentionOperator(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if model.grid_physics_refiner is None:
            raise ValueError("--freeze-backbone requires --physics-refiner")
        for parameter in model.grid_physics_refiner.parameters():
            parameter.requires_grad = True

    train = load_split(
        args.data_dir / f"{args.dataset_prefix}_train_000.parquet",
        8000,
    ).to(device)
    dev = load_split(
        args.data_dir / f"{args.dataset_prefix}_dev.parquet",
        1000,
    ).to(device)
    test = load_split(
        args.data_dir / f"{args.dataset_prefix}_test.parquet",
        1000,
    ).to(device)
    truth_pde = float(
        normalized_pde_residual(
            test[..., 0],
            args.equation,
            args.pde_beta,
        )[1]
    )

    initial_dev = evaluate(
        model,
        dev,
        device,
        args.equation,
        args.pde_beta,
    )
    print(f"initial dev={initial_dev} truth_pde={truth_pde:.6f}", flush=True)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=1e-7,
    )
    best_score = (
        initial_dev["l2"] + args.score_weight * initial_dev["pde_relative_rms"]
    )
    best_residual = initial_dev["pde_relative_rms"]
    data_budget = initial_dev["l2"] * (1.0 + args.max_l2_increase)
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.time()
    model.train()
    for step in range(args.steps):
        indices = torch.randint(0, train.size(0), (args.batch_size,), device=device)
        target = train[indices, :, :, 0]
        prediction = model.forward_grid(target[:, :, 0])
        data_loss = torch.mean((prediction - target) ** 2)
        pde_loss, pde_relative = normalized_pde_residual(
            prediction,
            args.equation,
            args.pde_beta,
        )
        ramp = min(1.0, (step + 1) / max(args.physics_warmup, 1))
        loss = data_loss + args.physics_weight * ramp * pde_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            dev_metrics = evaluate(
                model,
                dev,
                device,
                args.equation,
                args.pde_beta,
            )
            score = (
                dev_metrics["l2"]
                + args.score_weight * dev_metrics["pde_relative_rms"]
            )
            point = {
                "step": step + 1,
                "total_loss": float(loss.detach()),
                "data_loss": float(data_loss.detach()),
                "pde_loss": float(pde_loss.detach()),
                "pde_relative_rms": float(pde_relative.detach()),
                **{f"dev_{key}": value for key, value in dev_metrics.items()},
                "score": score,
                "physics_ramp": ramp,
                "data_budget": data_budget,
                "within_budget": dev_metrics["l2"] <= data_budget,
            }
            history.append(point)
            print(
                f"step={step + 1} data={point['data_loss']:.6f} "
                f"pde_rel={point['pde_relative_rms']:.6f} "
                f"dev_l2={dev_metrics['l2']:.6f} "
                f"dev_pde={dev_metrics['pde_relative_rms']:.6f} "
                f"score={score:.6f}",
                flush=True,
            )
            if (
                dev_metrics["l2"] <= data_budget
                and dev_metrics["pde_relative_rms"] < best_residual
            ):
                best_residual = dev_metrics["pde_relative_rms"]
                best_score = score
                best_step = step + 1
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_dev = evaluate(
        model,
        dev,
        device,
        args.equation,
        args.pde_beta,
    )
    final_test = evaluate(
        model,
        test,
        device,
        args.equation,
        args.pde_beta,
    )
    result = {
        "source_checkpoint": str(args.source_checkpoint),
        "parameter_count": count_parameters(model),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "best_step": best_step,
        "best_score": best_score,
        "best_residual": best_residual,
        "truth_pde_relative_rms": truth_pde,
        "initial_dev": initial_dev,
        "dev": final_dev,
        "test": final_test,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "physics_weight": args.physics_weight,
            "physics_warmup": args.physics_warmup,
            "score_weight": args.score_weight,
            "max_l2_increase": args.max_l2_increase,
            "seed": args.seed,
            "equation": args.equation,
            "pde_beta": args.pde_beta,
            "physics_refiner": args.physics_refiner,
            "refiner_hidden": args.refiner_hidden,
            "freeze_backbone": args.freeze_backbone,
        },
        "training_seconds": time.time() - started,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "operator": asdict(cfg),
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
