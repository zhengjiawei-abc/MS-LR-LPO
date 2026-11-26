"""Audit boundary, interior, high-order, and spectral PDE residuals."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from baselines_1d_burgers import FNO1d
from burgers_common import NU, T_FINAL, load_split
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from pde_residual_1d import pde_derivative_fields


EXP = ROOT / "experiments"
OUT = ROOT / "analysis_plots_clean" / "residual_credibility.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_subset(
    residual: torch.Tensor,
    denominator: torch.Tensor,
    x_slice: slice,
    t_slice: slice,
) -> float:
    numerator = residual[:, x_slice, t_slice].square().mean(dim=(1, 2))
    scale = denominator[:, x_slice, t_slice].mean(dim=(1, 2))
    return float((numerator / scale.clamp_min(1.0e-12)).mean().sqrt())


def original_residual_parts(
    fields: torch.Tensor,
    equation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        1.0,
    )
    if equation == "burgers":
        denominator = (
            u_t.square() + (fields * u_x).square() + (NU * u_xx).square()
        )
    else:
        denominator = u_t.square() + u_x.square()
    return residual, denominator


def high_order_relative(fields: torch.Tensor, equation: str) -> float:
    dx = 1.0 / fields.size(1)
    dt = T_FINAL / (fields.size(2) - 1)
    inter = fields[:, 2:-2, 2:-2]
    u_t = (
        fields[:, 2:-2, :-4]
        - 8.0 * fields[:, 2:-2, 1:-3]
        + 8.0 * fields[:, 2:-2, 3:-1]
        - fields[:, 2:-2, 4:]
    ) / (12.0 * dt)
    u_x = (
        fields[:, 4:, 2:-2]
        - 8.0 * fields[:, 3:-1, 2:-2]
        + 8.0 * fields[:, 1:-3, 2:-2]
        - fields[:, :-4, 2:-2]
    ) / (12.0 * dx)
    u_xx = (
        -fields[:, 4:, 2:-2]
        + 16.0 * fields[:, 3:-1, 2:-2]
        - 30.0 * inter
        + 16.0 * fields[:, 1:-3, 2:-2]
        - fields[:, :-4, 2:-2]
    ) / (12.0 * dx * dx)
    if equation == "burgers":
        residual = u_t + inter * u_x - NU * u_xx
        denominator = u_t.square() + (inter * u_x).square() + (NU * u_xx).square()
    else:
        residual = u_t + u_x
        denominator = u_t.square() + u_x.square()
    ratio = residual.square().mean(dim=(1, 2)) / denominator.mean(dim=(1, 2))
    return float(ratio.clamp_min(1.0e-12).mean().sqrt())


def spectral_high_frequency(fields: torch.Tensor) -> float:
    spectrum = torch.fft.rfft2(fields, dim=(1, 2)).abs().square()
    nx = fields.size(1)
    cutoff = max(nx // 8, 1)
    low = spectrum[:, :cutoff].sum(dim=(1, 2))
    total = spectrum.sum(dim=(1, 2)).clamp_min(1.0e-12)
    return float(((total - low) / total).mean())


def residual_report(fields: torch.Tensor, equation: str) -> dict:
    residual, denominator = original_residual_parts(fields, equation)
    boundary = fmean(
        [
            relative_subset(
                residual,
                denominator,
                slice(None),
                slice(0, 1),
            ),
            relative_subset(
                residual,
                denominator,
                slice(None),
                slice(-1, None),
            ),
        ]
    )
    return {
        "overall": relative_subset(
            residual,
            denominator,
            slice(None),
            slice(None),
        ),
        "boundary": boundary,
        "interior": relative_subset(residual, denominator, slice(None), slice(2, -2)),
        "high_order": high_order_relative(fields, equation),
        "high_frequency_energy_fraction": spectral_high_frequency(fields),
    }


def load_pg(path: Path, fields: torch.Tensor) -> torch.Tensor:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    valid = set(OperatorConfig.__dataclass_fields__)
    operator = {
        key: value
        for key, value in checkpoint["operator"].items()
        if key in valid
    }
    model = CoordAttentionOperator(OperatorConfig(**operator)).to(DEVICE).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    outputs = []
    with torch.no_grad():
        for start in range(0, fields.size(0), 64):
            outputs.append(model.forward_grid(fields[start : start + 64, :, 0, 0]))
    return torch.cat(outputs, dim=0)


def load_pino(path: Path, fields: torch.Tensor) -> torch.Tensor:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = FNO1d(
        config["modes"],
        config["width"],
        config["layers"],
        fields.size(2),
    ).to(DEVICE).eval()
    model.load_state_dict(checkpoint["state_dict"])
    outputs = []
    with torch.no_grad():
        for start in range(0, fields.size(0), 64):
            outputs.append(model(fields[start : start + 64, :, 0, 0]))
    return torch.cat(outputs, dim=0)


def summarize_model(
    fields: torch.Tensor,
    predictions: list[torch.Tensor],
    equation: str,
) -> dict:
    reports = [residual_report(prediction, equation) for prediction in predictions]
    result = {}
    for key in reports[0]:
        values = [entry[key] for entry in reports]
        result[key] = {
            "values": values,
            "mean": fmean(values),
        }
    return result


def main() -> None:
    burgers = load_split(
        ROOT
        / "datasets"
        / "pdebench_burgers_official"
        / "burgers_nu0.01_test.parquet",
        1000,
    ).to(DEVICE)
    advection = load_split(
        ROOT
        / "datasets"
        / "pdebench_advection_beta1_official"
        / "advection_beta1_test.parquet",
        1000,
    ).to(DEVICE)

    burgers_pg = [
        load_pg(
            EXP
            / "pg_physics_hpo"
            / f"refiner_h18_k11_w03_deriv2k5_4k_budget02_s{seed}"
            / "model.pt",
            burgers,
        )
        for seed in (7, 11, 23)
    ]
    burgers_pino = [
        load_pino(
            EXP / "pino_formal" / "burgers" / f"w0001_s{seed}_24k" / "model.pt",
            burgers,
        )
        for seed in (7, 11, 23)
    ]
    advection_pg = [
        load_pg(
            EXP
            / "pg_advection_output_refiner"
            / f"w0008_budget30_s{seed}_4k"
            / "model.pt",
            advection,
        )
        for seed in (7, 11, 23)
    ]
    advection_pino = [
        load_pino(
            EXP
            / "pino_formal"
            / "advection_beta1"
            / f"w0005_s{seed}_24k"
            / "model.pt",
            advection,
        )
        for seed in (7, 11, 23)
    ]

    result = {
        "scope": (
            "Official test split; boundary is t=0 and t=T, interior excludes "
            "two time cells at both ends; high-order uses fourth-order central "
            "differences."
        ),
        "Burgers": {
            "truth": residual_report(burgers[..., 0], "burgers"),
            "pg_derivative_refiner": summarize_model(
                burgers,
                burgers_pg,
                "burgers",
            ),
            "pino": summarize_model(
                burgers,
                burgers_pino,
                "burgers",
            ),
        },
        "Advection beta=1": {
            "truth": residual_report(advection[..., 0], "advection"),
            "pg_output_refiner": summarize_model(
                advection,
                advection_pg,
                "advection",
            ),
            "pino": summarize_model(
                advection,
                advection_pino,
                "advection",
            ),
        },
    }
    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
