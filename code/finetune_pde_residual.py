"""Fine-tune the low-rank model with a normalized Burgers PDE residual."""
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
OUTPUT = RUNS / "latent_pde_physics_joint"
STEPS = 4000
BATCH = 64
LR = 5e-5
PDE_WEIGHT = 5e-4
DEVICE = torch.device("cuda")


def pde_residual_loss(fields: torch.Tensor):
    dx = 1.0 / NX
    dt = T_FINAL / (NT - 1)
    u_t = torch.gradient(fields, spacing=(dt,), dim=2)[0]
    u_x = (torch.roll(fields, -1, 1) - torch.roll(fields, 1, 1)) / (2 * dx)
    u_xx = (
        torch.roll(fields, -1, 1)
        - 2 * fields
        + torch.roll(fields, 1, 1)
    ) / (dx * dx)
    residual = u_t + fields * u_x - NU * u_xx
    numerator = residual.square().mean(dim=(1, 2))
    denominator = (
        u_t.square()
        + (fields * u_x).square()
        + (NU * u_xx).square()
    ).mean(dim=(1, 2)).clamp_min(1e-12)
    relative = numerator / denominator
    return relative.mean(), relative.sqrt().mean()


def main() -> None:
    set_seed(7)
    checkpoint = torch.load(SOURCE, map_location=DEVICE, weights_only=False)
    model = CoordAttentionOperator(
        OperatorConfig(**checkpoint["operator"])
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    train = load_split(
        DATA / "burgers_nu0.01_train_000.parquet", 8000
    ).to(DEVICE)
    dev = load_split(DATA / "burgers_nu0.01_dev.parquet", 1000).to(DEVICE)
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)

    model.eval()
    initial_pred = predict_full(model, dev, DEVICE)
    initial = {
        "dev_l2": relative_l2(initial_pred, dev[..., 0]),
        "dev_pde_relative": float(pde_residual_loss(initial_pred)[1]),
    }
    print(f"initial={initial}", flush=True)

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
        pde_loss, _ = pde_residual_loss(prediction)
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
            dev_pde = float(pde_residual_loss(dev_pred)[1])
            score = dev_l2 + 0.005 * dev_pde
            print(
                f"step={step + 1} data={data_loss.item():.6f} "
                f"pde={pde_loss.item():.6f} dev_l2={dev_l2:.6f} "
                f"dev_pde={dev_pde:.6f} score={score:.6f}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    final_pred = predict_full(model, test, DEVICE)
    final = {
        "test_l2": relative_l2(final_pred, test[..., 0]),
        "test_pde_relative": float(pde_residual_loss(final_pred)[1]),
    }
    result = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial": initial,
        "final": final,
        "pde_weight": PDE_WEIGHT,
        "steps": STEPS,
        "training_seconds": time.time() - started,
        "config": checkpoint["operator"],
    }
    print(json.dumps(result, indent=2), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "operator": checkpoint["operator"]},
        OUTPUT / "model.pt",
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
