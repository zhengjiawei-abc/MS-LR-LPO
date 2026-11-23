"""Estimate advection beta from clean or noisy space-time observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from burgers_common import NT, NX, T_FINAL, load_split, relative_l2


ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    0.2: ("advection_beta02", "pdebench_advection_beta02_official"),
    0.7: ("advection_beta07", "pdebench_advection_beta07_official"),
    1.0: ("advection_beta1", "pdebench_advection_beta1_official"),
    2.0: ("advection_beta20", "pdebench_advection_beta20_official"),
    4.0: ("advection_beta40", "pdebench_advection_beta40_official"),
    7.0: ("advection_beta70", "pdebench_advection_beta70_official"),
}


def transported_field(
    initial: torch.Tensor,
    beta: float,
    time_indices: torch.Tensor,
) -> torch.Tensor:
    batch = initial.size(0)
    times = torch.linspace(0.0, T_FINAL, NT, device=initial.device)[time_indices]
    base = torch.arange(NX, device=initial.device).float()
    position = (
        base[None, :, None] - beta * times[None, None, :] * NX
    ).remainder(NX)
    index0 = torch.floor(position).long()
    index1 = (index0 + 1).remainder(NX)
    fraction = position - index0.float()
    source = initial.unsqueeze(-1).expand(-1, -1, time_indices.numel())
    lower = torch.gather(source, 1, index0.expand(batch, -1, -1))
    upper = torch.gather(source, 1, index1.expand(batch, -1, -1))
    return lower + fraction * (upper - lower)


@torch.no_grad()
def estimate_beta(
    fields: torch.Tensor,
    candidates: torch.Tensor,
    time_indices: torch.Tensor,
    noise: float,
) -> torch.Tensor:
    initial = fields[:, :, 0, 0]
    target = fields[:, :, time_indices, 0]
    if noise > 0:
        target = target + torch.randn_like(target) * target.std() * noise
    errors = []
    for beta in candidates:
        prediction = transported_field(initial, float(beta), time_indices)
        errors.append(((prediction - target) ** 2).mean(dim=(1, 2)))
    error_matrix = torch.stack(errors, dim=1)
    return candidates[error_matrix.argmin(dim=1)]


def parse_noise_levels(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--noise-levels", default="0,0.01,0.05,0.1")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=8.0)
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = torch.linspace(
        args.beta_min,
        args.beta_max,
        args.grid_size,
        device=device,
    )
    if args.partial:
        time_indices = torch.tensor([8, 19, 31, 43, 58, 71, 83, 97], device=device)
        observation_name = "8_snapshots"
    else:
        time_indices = torch.arange(1, NT, device=device)
        observation_name = "full_trajectory"

    results = {
        "observation": observation_name,
        "beta_range": [args.beta_min, args.beta_max],
        "grid_size": args.grid_size,
        "samples_per_beta": args.samples,
        "noise_results": [],
    }
    for noise in parse_noise_levels(args.noise_levels):
        per_parameter = {}
        errors = []
        for beta, (prefix, directory) in DATASETS.items():
            path = ROOT / "datasets" / directory / f"{prefix}_test.parquet"
            fields = load_split(path, args.samples).to(device)
            estimate = estimate_beta(fields, candidates, time_indices, noise)
            error = (estimate - beta).abs()
            errors.append(error)
            per_parameter[str(beta)] = {
                "mae": float(error.mean()),
                "within_5pct": float((error / beta <= 0.05).float().mean()),
            }
        error = torch.cat(errors)
        true_values = torch.cat(
            [torch.full((args.samples,), beta, device=device) for beta in DATASETS]
        )
        relative = error / true_values
        results["noise_results"].append(
            {
                "noise": noise,
                "mae": float(error.mean()),
                "rmse": float(torch.sqrt(error.square().mean())),
                "mape": float(relative.mean()),
                "within_5pct": float((relative <= 0.05).float().mean()),
                "per_parameter": per_parameter,
            }
        )
        print(json.dumps(results["noise_results"][-1], indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
