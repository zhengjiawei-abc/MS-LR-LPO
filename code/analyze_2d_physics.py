"""Evaluate corrected 2D diffusion-reaction and NS momentum residuals."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from analyze_2d_models import load_model, predict
from operator_2d_common import (
    load_npz_split,
    load_spec,
    ns_momentum_divergence_relative,
    pde_residual_2d_relative,
    space_time_coordinates,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = ROOT / "analysis_plots_clean" / "physics_2d_audit.json"


def mean_relative(values: torch.Tensor) -> float:
    return float(values.mean())


def diffusion_reaction() -> dict:
    dataset = ROOT / "datasets" / "pdebench_diff_react_2d_t10"
    spec = load_spec(dataset / "metadata.json")
    _, target = load_npz_split(dataset / "test.npz")
    coordinates = space_time_coordinates(spec, DEVICE)
    target = target.float()
    result = {
        "truth": mean_relative(
            pde_residual_2d_relative(target.to(DEVICE), coordinates)
        )
    }
    checkpoints = {
        "pg": EXP
        / "outputs_2d"
        / "pg2d_diff_react_t10_seed7_fixed"
        / "model.pt",
        "fno": EXP
        / "outputs_2d"
        / "fno2d_diff_react_t10_seed7"
        / "model.pt",
    }
    inputs, _ = load_npz_split(dataset / "test.npz")
    for name, path in checkpoints.items():
        model, _ = load_model(path, spec, DEVICE)
        prediction = predict(model, inputs, DEVICE)
        result[name] = mean_relative(
            pde_residual_2d_relative(prediction.to(DEVICE), coordinates)
        )
    return result


def ns_particles(test_inputs: torch.Tensor) -> torch.Tensor:
    source = (
        ROOT
        / "datasets"
        / "pdebench_raw"
        / "ns_incom_inhom_2d_512-0.h5"
    )
    with h5py.File(source, "r") as handle:
        particles = handle["particles"][
            3,
            :,
            ::8,
            ::8,
            0,
        ]
    windows = []
    for start in range(0, particles.shape[0] - 41 + 1, 20):
        windows.append(
            particles[start : start + 41].transpose(1, 2, 0)
        )
    result = torch.from_numpy(np.stack(windows).astype(np.float32))
    if result.size(0) != test_inputs.size(0):
        raise RuntimeError(
            f"particle windows {result.size(0)} != test {test_inputs.size(0)}"
        )
    return result


def incompressible_ns() -> dict:
    dataset = ROOT / "datasets" / "pdebench_ns_incom_2d"
    spec = load_spec(dataset / "metadata.json")
    inputs, target = load_npz_split(dataset / "test.npz")
    coordinates = space_time_coordinates(spec, DEVICE)
    force = inputs[:, 2:4].float()
    particles = ns_particles(inputs)
    result = {
        "truth": mean_relative(
            ns_momentum_divergence_relative(
                target.float().to(DEVICE),
                coordinates,
                force=force.to(DEVICE),
                particles=particles.to(DEVICE),
            )
        )
    }
    checkpoints = {
        "pg": EXP / "outputs_2d" / "ns_seeds_fixed" / "pg2d_seed7" / "model.pt",
        "fno": EXP / "outputs_2d" / "ns_fno_seed7" / "model.pt",
    }
    for name, path in checkpoints.items():
        model, _ = load_model(path, spec, DEVICE)
        prediction = predict(model, inputs, DEVICE)
        result[name] = mean_relative(
            ns_momentum_divergence_relative(
                prediction.to(DEVICE),
                coordinates,
                force=force.to(DEVICE),
                particles=particles.to(DEVICE),
            )
        )
    return result


def main() -> None:
    result = {
        "diffusion_reaction": diffusion_reaction(),
        "incompressible_ns": incompressible_ns(),
        "notes": {
            "diffusion_reaction": (
                "PDEBench form: u_t=Du*lap(u)+u-u^3-k-v, "
                "v_t=Dv*lap(v)+u-v; Du=1e-3, Dv=5e-3, k=5e-3."
            ),
            "incompressible_ns": (
                "Divergence of momentum equation with pressure eliminated; "
                "source-file particle density and force are used."
            ),
        },
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


EXP = ROOT / "experiments"


if __name__ == "__main__":
    main()
