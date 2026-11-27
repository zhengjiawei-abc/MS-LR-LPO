"""Jointly fine-tune the Burgers operator on Nu=0.01 and Nu=0.1."""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from burgers_common import NU, count_parameters, load_split, relative_l2, set_seed
from coord_attention_operator import CoordAttentionOperator, OperatorConfig


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = (
    ROOT
    / "experiments"
    / "outputs_burgers_dataset"
    / "official_n8000"
    / "latent_pde_all_lowrank_joint"
    / "model.pt"
)
DATA_001 = ROOT / "datasets" / "pdebench_burgers_official"
DATA_01 = ROOT / "datasets" / "external" / "burgers_nu0.1"


def load_npz(path: Path, limit: int) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        fields = data["fields"][:limit]
    return torch.from_numpy(fields).float()


def predict(
    model: CoordAttentionOperator,
    fields: torch.Tensor,
    viscosity: float,
) -> torch.Tensor:
    model.eval()
    outputs = []
    label = torch.full(
        (1,),
        viscosity,
        device=fields.device,
        dtype=fields.dtype,
    )
    with torch.no_grad():
        for start in range(0, fields.size(0), 64):
            batch = fields[start : start + 64]
            labels = label.expand(batch.size(0))
            outputs.append(
                model.forward_grid(batch[:, :, 0, 0], viscosity=labels)
            )
    model.train()
    return torch.cat(outputs, dim=0)


def evaluate(
    model: CoordAttentionOperator,
    fields_001: torch.Tensor,
    fields_01: torch.Tensor,
) -> dict:
    first = predict(model, fields_001, NU)
    second = predict(model, fields_01, 0.1)
    l2_001 = relative_l2(first, fields_001[..., 0])
    l2_01 = relative_l2(second, fields_01[..., 0])
    return {
        "nu0.01_l2": l2_001,
        "nu0.1_l2": l2_01,
        "score": 0.5 * (l2_001 + l2_01),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "burgers_multiviscosity"
            / "nu001_01_s7_6k"
        ),
    )
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.source_checkpoint,
        map_location=device,
        weights_only=False,
    )
    valid = set(OperatorConfig.__dataclass_fields__)
    operator = {
        key: value
        for key, value in checkpoint["operator"].items()
        if key in valid
    }
    operator["viscosity_conditioning"] = True
    config = OperatorConfig(**operator)
    model = CoordAttentionOperator(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    teacher = CoordAttentionOperator(config).to(device).eval()
    teacher.load_state_dict(checkpoint["state_dict"], strict=False)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    train_001 = load_split(DATA_001 / "burgers_nu0.01_train_000.parquet", 8000).to(
        device
    )
    train_01 = load_npz(DATA_01 / "train.npz", 2000).to(device)
    dev_001 = load_split(DATA_001 / "burgers_nu0.01_dev.parquet", 1000).to(device)
    dev_01 = load_npz(DATA_01 / "dev.npz", 1000).to(device)
    test_001 = load_split(DATA_001 / "burgers_nu0.01_test.parquet", 1000).to(device)
    test_01 = load_npz(DATA_01 / "test.npz", 1000).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=1.0e-6,
    )
    best_score = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.time()
    model.train()

    for step in range(args.steps):
        half = args.batch_size // 2
        index_001 = torch.randint(0, train_001.size(0), (half,), device=device)
        index_01 = torch.randint(0, train_01.size(0), (half,), device=device)
        target = torch.cat(
            [train_001[index_001], train_01[index_01]],
            dim=0,
        )
        viscosity = torch.cat(
            [
                torch.full((half,), NU, device=device),
                torch.full((args.batch_size - half,), 0.1, device=device),
            ]
        )
        prediction = model.forward_grid(
            target[:, :, 0, 0],
            viscosity=viscosity,
        )
        data_001 = torch.mean(
            (prediction[:half] - target[:half, :, :, 0]) ** 2
        )
        data_01 = torch.mean(
            (prediction[half:] - target[half:, :, :, 0]) ** 2
        )
        with torch.no_grad():
            teacher_001 = teacher.forward_grid(
                target[:half, :, 0, 0],
                viscosity=torch.full((half,), NU, device=device),
            )
        distillation = torch.mean((prediction[:half] - teacher_001) ** 2)
        data_loss = 0.5 * (data_001 + data_01)
        loss = data_loss + args.distill_weight * distillation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = evaluate(model, dev_001, dev_01)
            point = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "data_nu0.01": float(data_001.detach()),
                "data_nu0.1": float(data_01.detach()),
                "distillation": float(distillation.detach()),
                **metrics,
            }
            history.append(point)
            print(point, flush=True)
            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best_step = step + 1
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_001, test_01)
    result = {
        "source_checkpoint": str(args.source_checkpoint),
        "parameter_count": count_parameters(model),
        "best_step": best_step,
        "best_dev_score": best_score,
        "test": test_metrics,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "distill_weight": args.distill_weight,
        },
        "training_seconds": time.time() - started,
        "history": history,
    }
    torch.save(
        {
            "state_dict": best_state,
            "operator": asdict(config),
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
