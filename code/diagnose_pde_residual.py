"""Measure Burgers PDE residuals for truth and the best operator checkpoint."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from burgers_common import NU, NT, NX, T_FINAL, load_split, relative_l2
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from train_coord_attention import predict_full


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
RUN = (
    ROOT
    / "experiments"
    / "outputs_burgers_dataset"
    / "official_n8000"
    / "latent_pde_all_lowrank_joint"
)
DEVICE = torch.device("cuda")


def derivatives(u: torch.Tensor):
    dx = 1.0 / NX
    dt = T_FINAL / (NT - 1)
    u_t = torch.gradient(u, spacing=(dt,), dim=2)[0]
    u_x = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
    u_xx = (
        torch.roll(u, -1, dims=1)
        - 2 * u
        + torch.roll(u, 1, dims=1)
    ) / (dx * dx)
    return u_t, u_x, u_xx


def residual_metrics(u: torch.Tensor):
    u_t, u_x, u_xx = derivatives(u)
    residual = u_t + u * u_x - NU * u_xx
    numerator = torch.mean(residual.square(), dim=(1, 2))
    denominator = torch.mean(
        u_t.square() + (u * u_x).square() + (NU * u_xx).square(),
        dim=(1, 2),
    ).clamp_min(1e-12)
    relative = (numerator / denominator).sqrt()
    raw_rms = numerator.sqrt()
    return {
        "raw_rms_mean": float(raw_rms.mean()),
        "raw_rms_median": float(raw_rms.median()),
        "relative_rms_mean": float(relative.mean()),
        "relative_rms_median": float(relative.median()),
    }


def main() -> None:
    checkpoint = torch.load(RUN / "model.pt", map_location=DEVICE, weights_only=False)
    model = CoordAttentionOperator(
        OperatorConfig(**checkpoint["operator"])
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)
    target = test[..., 0]
    prediction = predict_full(model, test, DEVICE)
    result = {
        "relative_l2": relative_l2(prediction, target),
        "truth": residual_metrics(target),
        "prediction": residual_metrics(prediction),
    }
    print(json.dumps(result, indent=2), flush=True)
    (RUN / "pde_residual_diagnostics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
