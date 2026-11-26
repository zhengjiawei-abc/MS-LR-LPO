"""Paired test-set uncertainty for the four primary PG-versus-PINO claims."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from analyze_residual_credibility import load_pg, load_pino
from burgers_common import load_split
from pde_residual_1d import normalized_pde_residual_per_sample


EXP = ROOT / "experiments"
OUTPUT = ROOT / "analysis_plots_clean" / "paired_statistics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_sample_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.linalg.vector_norm(
        (prediction - target).flatten(1),
        dim=1,
    )
    scale = torch.linalg.vector_norm(
        target.flatten(1),
        dim=1,
    ).clamp_min(1.0e-8)
    return error / scale


def bootstrap_interval(
    differences: torch.Tensor,
    samples: int = 20000,
) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    values = differences.detach().cpu()
    indices = torch.randint(
        0,
        values.numel(),
        (samples, values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    return {
        "mean_difference": float(values.mean()),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
    }


def compare(
    pg: torch.Tensor,
    pino: torch.Tensor,
) -> dict:
    difference = pg - pino
    paired = ttest_rel(
        pg.detach().cpu().numpy(),
        pino.detach().cpu().numpy(),
    )
    signed_rank = wilcoxon(
        pg.detach().cpu().numpy(),
        pino.detach().cpu().numpy(),
        zero_method="wilcox",
    )
    return {
        **bootstrap_interval(difference),
        "paired_t_p": float(paired.pvalue),
        "wilcoxon_p": float(signed_rank.pvalue),
        "pg_better_fraction": float((difference < 0).float().mean()),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    result = {}
    previous = 0.0
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * value)
        adjusted = max(previous, adjusted)
        result[name] = adjusted
        previous = adjusted
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

    burgers_pg = load_pg(
        EXP
        / "pg_physics_hpo"
        / "refiner_h18_k11_w03_deriv2k5_4k_budget02_s7"
        / "model.pt",
        burgers,
    )
    burgers_pino = load_pino(
        EXP / "pino_formal" / "burgers" / "w0001_s7_24k" / "model.pt",
        burgers,
    )
    advection_pg = load_pg(
        EXP
        / "pg_advection_output_refiner"
        / "w0008_budget30_s7_4k"
        / "model.pt",
        advection,
    )
    advection_pino = load_pino(
        EXP
        / "pino_formal"
        / "advection_beta1"
        / "w0005_s7_24k"
        / "model.pt",
        advection,
    )

    comparisons = {
        "Burgers/L2": compare(
            per_sample_l2(burgers_pg, burgers[..., 0]),
            per_sample_l2(burgers_pino, burgers[..., 0]),
        ),
        "Burgers/residual": compare(
            normalized_pde_residual_per_sample(burgers_pg, "burgers", 1.0),
            normalized_pde_residual_per_sample(burgers_pino, "burgers", 1.0),
        ),
        "Advection beta=1/L2": compare(
            per_sample_l2(advection_pg, advection[..., 0]),
            per_sample_l2(advection_pino, advection[..., 0]),
        ),
        "Advection beta=1/residual": compare(
            normalized_pde_residual_per_sample(
                advection_pg,
                "advection",
                1.0,
            ),
            normalized_pde_residual_per_sample(
                advection_pino,
                "advection",
                1.0,
            ),
        ),
    }
    adjusted = holm(
        {name: value["paired_t_p"] for name, value in comparisons.items()}
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    result = {
        "protocol": (
            "Seed-7 common 1000-sample test split. Negative differences mean "
            "PG has lower error. Bootstrap resamples samples; Holm correction "
            "covers four primary claims."
        ),
        "comparisons": comparisons,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
