"""Compile and plot DeepONet query-point supervision across 1D tasks."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CROSS = ROOT / "experiments" / "outputs_cross_pde"
BURGERS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
HIGH = CROSS / "deeponet_high_supervision"

INK = "#17212b"
MUTED = "#64748b"
GRID = "#d7dee8"
LOW = "#94a3b8"
HIGH_COLOR = "#087f8c"

RUNS = {
    "Burgers": {
        "low": [
            BURGERS / f"deeponet_pm501_s{seed}_24k" / "metrics.json"
            for seed in (7, 11, 23)
        ],
        "high": [
            BURGERS / f"deeponet_pm501_dense_s{seed}_24k" / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
    "Advection beta=1": {
        "low": [
            CROSS / "full" / "advection_beta1" / "deeponet" / "metrics.json",
            CROSS / "seeds" / "advection_beta1" / "deeponet_s11" / "metrics.json",
            CROSS / "seeds" / "advection_beta1" / "deeponet_s23" / "metrics.json",
        ],
        "high": [
            HIGH
            / "advection_beta1"
            / f"deeponet_high4096_s{seed}_high4096"
            / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
    "Diffusion-sorption": {
        "low": [
            CROSS / "full" / "diff_sorp" / "deeponet" / "metrics.json",
            CROSS / "seeds" / "diff_sorp" / "deeponet_s11" / "metrics.json",
            CROSS / "seeds" / "diff_sorp" / "deeponet_s23" / "metrics.json",
        ],
        "high": [
            HIGH
            / "diff_sorp"
            / f"deeponet_high4096_s{seed}_high4096"
            / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
}


def read_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(paths: list[Path], query_points: int) -> dict:
    metrics = [read_metrics(path) for path in paths]
    values = [float(item["test_relative_l2"]) for item in metrics]
    return {
        "query_points": query_points,
        "test_values": values,
        "test_mean": fmean(values),
        "test_std": stdev(values),
        "training_seconds_mean": fmean(
            float(item["training_seconds"]) for item in metrics
        ),
        "parameters": int(metrics[0]["parameter_count"]),
    }


def main() -> None:
    summary = {}
    for task, paths in RUNS.items():
        summary[task] = {
            "low_supervision": summarize(paths["low"], 256),
            "high_supervision": summarize(paths["high"], 4096),
        }
    (OUT / "deeponet_supervision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    labels = list(summary)
    low = np.array(
        [summary[task]["low_supervision"]["test_mean"] for task in labels]
    )
    high = np.array(
        [summary[task]["high_supervision"]["test_mean"] for task in labels]
    )
    low_std = np.array(
        [summary[task]["low_supervision"]["test_std"] for task in labels]
    )
    high_std = np.array(
        [summary[task]["high_supervision"]["test_std"] for task in labels]
    )
    x = np.arange(len(labels))

    fig, axis = plt.subplots(figsize=(12, 5.8))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    width = 0.35
    low_bars = axis.bar(
        x - width / 2,
        low,
        width,
        yerr=low_std,
        capsize=4,
        color=LOW,
        label="256 query points",
    )
    high_bars = axis.bar(
        x + width / 2,
        high,
        width,
        yerr=high_std,
        capsize=4,
        color=HIGH_COLOR,
        label="4096 query points",
    )
    axis.set_yscale("log")
    axis.set_ylim(0.011, 0.95)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Relative Test L2 (log scale, lower is better)")
    axis.set_title(
        "Same parameters and 24K optimization steps; only query points change",
        fontsize=10.5,
        pad=12,
    )
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )

    for index, (low_bar, high_bar) in enumerate(zip(low_bars, high_bars)):
        reduction = (1.0 - high[index] / low[index]) * 100.0
        axis.text(
            low_bar.get_x() + low_bar.get_width() / 2,
            low[index] * 1.28,
            f"{low[index]:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )
        axis.text(
            high_bar.get_x() + high_bar.get_width() / 2,
            high[index] * 1.28,
            f"{high[index]:.4f}\n({reduction:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=HIGH_COLOR,
            fontweight="bold",
        )

    fig.suptitle(
        "DeepONet supervision ablation across 1D tasks",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "deeponet_supervision.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
