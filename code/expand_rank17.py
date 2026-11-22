"""Expand rank16 decoder blocks to rank17 without changing the function."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import torch

from burgers_common import NU, NT, NX, T_FINAL, load_split, relative_l2, set_seed
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from train_coord_attention import predict_full


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
SOURCE = RUNS / "latent_pde_all_lowrank_joint" / "model.pt"
OUTPUT = RUNS / "latent_pde_501k_expanded"
STEPS = 4000
BATCH = 64
LR = 5e-5
PDE_WEIGHT = 5e-4
DEVICE = torch.device("cuda")


def pde_metric(fields: torch.Tensor):
    dx = 1.0 / NX
    dt = T_FINAL / (NT - 1)
    u_t = torch.gradient(fields, spacing=(dt,), dim=2)[0]
    u_x = (torch.roll(fields, -1, 1) - torch.roll(fields, 1, 1)) / (2 * dx)
    u_xx = (
        torch.roll(fields, -1, 1) - 2 * fields + torch.roll(fields, 1, 1)
    ) / (dx * dx)
    residual = u_t + fields * u_x - NU * u_xx
    numerator = residual.square().mean(dim=(1, 2))
    denominator = (
        u_t.square()
        + (fields * u_x).square()
        + (NU * u_xx).square()
    ).mean(dim=(1, 2)).clamp_min(1e-12)
    return numerator / denominator


def expand_block(source, target) -> None:
    target.body[0].weight.data[:-1].copy_(source["body.0.weight"])
    target.body[0].weight.data[-1].zero_()
    target.body[1].weight.data[:, :-1].copy_(source["body.1.weight"])
    target.body[1].weight.data[:, -1].zero_()
    target.body[1].bias.data.copy_(source["body.1.bias"])
    target.body[3].weight.data[:-1].copy_(source["body.3.weight"])
    target.body[3].weight.data[-1].zero_()
    target.body[4].weight.data[:, :-1].copy_(source["body.4.weight"])
    target.body[4].weight.data[:, -1].zero_()
    target.body[4].bias.data.copy_(source["body.4.bias"])


def main() -> None:
    set_seed(7)
    checkpoint = torch.load(SOURCE, map_location=DEVICE, weights_only=False)
    operator = dict(checkpoint["operator"])
    operator["grid_lowrank_rank"] = 17
    model = CoordAttentionOperator(OperatorConfig(**operator)).to(DEVICE)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in checkpoint["state_dict"].items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    for index, block in enumerate(model.grid_decoder.blocks):
        prefix = f"grid_decoder.blocks.{index}."
        source = {
            "norm.weight": checkpoint["state_dict"][prefix + "norm.weight"],
            "norm.bias": checkpoint["state_dict"][prefix + "norm.bias"],
            "body.0.weight": checkpoint["state_dict"][prefix + "body.0.weight"],
            "body.1.weight": checkpoint["state_dict"][prefix + "body.1.weight"],
            "body.1.bias": checkpoint["state_dict"][prefix + "body.1.bias"],
            "body.3.weight": checkpoint["state_dict"][prefix + "body.3.weight"],
            "body.4.weight": checkpoint["state_dict"][prefix + "body.4.weight"],
            "body.4.bias": checkpoint["state_dict"][prefix + "body.4.bias"],
        }
        expand_block(source, block)
        block.norm.load_state_dict(
            {
                "weight": source["norm.weight"],
                "bias": source["norm.bias"],
            }
        )

    train = load_split(
        DATA / "burgers_nu0.01_train_000.parquet", 8000
    ).to(DEVICE)
    dev = load_split(DATA / "burgers_nu0.01_dev.parquet", 1000).to(DEVICE)
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)
    model.eval()
    initial_pred = predict_full(model, test, DEVICE)
    initial = {
        "test_l2": relative_l2(initial_pred, test[..., 0]),
        "test_pde": float(pde_metric(initial_pred).sqrt().mean()),
    }
    print(
        f"params={sum(p.numel() for p in model.parameters())} initial={initial}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STEPS,
        eta_min=1e-7,
    )
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    model.train()
    for step in range(STEPS):
        index = torch.randint(0, train.size(0), (BATCH,), device=DEVICE)
        target = train[index, :, :, 0]
        prediction = model.forward_grid(target[:, :, 0])
        data_loss = torch.mean((prediction - target) ** 2)
        pde_loss = pde_metric(prediction).mean()
        loss = data_loss + PDE_WEIGHT * pde_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 500 == 0:
            model.eval()
            dev_pred = predict_full(model, dev, DEVICE)
            dev_l2 = relative_l2(dev_pred, dev[..., 0])
            dev_pde = float(pde_metric(dev_pred).sqrt().mean())
            score = dev_l2 + 0.005 * dev_pde
            print(
                f"step={step + 1} data={data_loss.item():.6f} "
                f"pde={pde_loss.item():.6f} dev_l2={dev_l2:.6f} "
                f"dev_pde={dev_pde:.6f}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    test_pred = predict_full(model, test, DEVICE)
    result = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial": initial,
        "final": {
            "test_l2": relative_l2(test_pred, test[..., 0]),
            "test_pde": float(pde_metric(test_pred).sqrt().mean()),
        },
        "steps": STEPS,
        "training_seconds": time.time() - started,
        "config": operator,
    }
    print(json.dumps(result, indent=2), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "operator": operator}, OUTPUT / "model.pt")
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
