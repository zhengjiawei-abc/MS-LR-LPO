"""Jointly fine-tune the sub-FNO low-rank model for higher accuracy."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from burgers_common import load_split, relative_l2, set_seed
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from train_coord_attention import predict_full


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
BATCH = 64
LR = 2e-4
DEVICE = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=RUNS / "latent_pde_all_lowrank" / "model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS / "latent_pde_all_lowrank_joint",
    )
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def evaluate(model, fields):
    model.eval()
    value = relative_l2(predict_full(model, fields, DEVICE), fields[..., 0])
    model.train()
    return value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = torch.load(args.source, map_location=DEVICE, weights_only=False)
    model = CoordAttentionOperator(
        OperatorConfig(**checkpoint["operator"])
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    train = load_split(
        DATA / "burgers_nu0.01_train_000.parquet", 8000
    ).to(DEVICE)
    dev = load_split(DATA / "burgers_nu0.01_dev.parquet", 1000).to(DEVICE)
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)
    model.train()
    initial = {
        "dev": relative_l2(predict_full(model, dev, DEVICE), dev[..., 0]),
        "test": relative_l2(predict_full(model, test, DEVICE), test[..., 0]),
    }
    print(f"initial={initial}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    def lr_factor(step: int) -> float:
        warmup = 200
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 1e-6 / LR + (1.0 - 1e-6 / LR) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best_dev = initial["dev"]
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    for step in range(args.steps):
        index = torch.randint(0, train.size(0), (BATCH,), device=DEVICE)
        target = train[index, :, :, 0]
        prediction = model.forward_grid(target[:, :, 0])
        loss = torch.mean((prediction - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 500 == 0:
            dev_l2 = evaluate(model, dev)
            print(
                f"step={step + 1} loss={loss.item():.6f} dev={dev_l2:.6f}",
                flush=True,
            )
            if dev_l2 < best_dev:
                best_dev = dev_l2
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    final = {
        "dev": relative_l2(predict_full(model, dev, DEVICE), dev[..., 0]),
        "test": relative_l2(predict_full(model, test, DEVICE), test[..., 0]),
    }
    result = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial": initial,
        "final": final,
        "steps": args.steps,
        "seed": args.seed,
        "training_seconds": time.time() - started,
        "config": checkpoint["operator"],
    }
    print(json.dumps(result, indent=2), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "operator": checkpoint["operator"]},
        args.output_dir / "model.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
