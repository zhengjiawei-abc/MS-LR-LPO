"""Generate parameter-sweep and residual-branch analysis figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "experiments" / "outputs_cross_pde" / "advection_parameters"

INK = "#17212b"
MUTED = "#64748b"
GRID = "#d7dee8"
PG = "#087f8c"
FIXED = "#7aa6a9"
FNO = "#d97706"
DEEPONET = "#c2415d"
LIGHT = "#eef6f7"

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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def parameter_sweep() -> None:
    results = read_json(OUT / "advection_eight_parameter_summary.json")
    residual_summary = read_json(OUT / "residual_seed_summary.json")
    betas = [0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 7.0]
    fixed = [
        results[f"beta{beta:g}"]["fixed_transport"]
        for beta in betas
    ]
    residual_lookup = {
        beta: residual_summary[key]["test_mean"]
        for beta, key in ((2.0, "beta2"), (4.0, "beta4"), (7.0, "beta7"))
    }
    residual = [residual_lookup.get(beta) for beta in betas]
    fno = [
        results[f"beta{beta:g}"]["fno"]["test_mean"]
        for beta in betas
    ]
    deeponet = [
        results[f"beta{beta:g}"]["deeponet"]["test_mean"]
        for beta in betas
    ]
    best_pg = [
        fixed_value if residual_value is None else min(fixed_value, residual_value)
        for fixed_value, residual_value in zip(fixed, residual)
    ]

    x = np.arange(len(betas))
    fig, axis = plt.subplots(figsize=(12, 5.8))
    style_axis(axis)
    axis.set_yscale("log")
    axis.plot(
        x,
        fixed,
        marker="o",
        linewidth=2,
        color=FIXED,
        label="Fixed transport PG",
    )
    axis.plot(
        x,
        best_pg,
        marker="o",
        markersize=8,
        linewidth=3,
        color=PG,
        label="Best PG",
    )
    axis.plot(
        x,
        fno,
        marker="s",
        linewidth=2,
        color=FNO,
        label="Parameter-matched FNO",
    )
    axis.plot(
        x,
        deeponet,
        marker="^",
        linewidth=2,
        color=DEEPONET,
        label="DeepONet",
    )
    for index, value in enumerate(best_pg):
        axis.text(
            index,
            value * 0.72,
            f"{value:.4f}",
            ha="center",
            va="top",
            fontsize=8,
            color=PG,
            fontweight="bold",
        )
    axis.annotate(
        "FNO still wins at beta=7",
        xy=(7, fno[-1]),
        xytext=(5.8, 0.013),
        color=FNO,
        fontsize=9,
        arrowprops={"arrowstyle": "-|>", "color": FNO, "linewidth": 1.4},
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": GRID},
    )
    axis.set_xticks(x, [f"{beta:g}" for beta in betas])
    axis.set_xlabel("Advection coefficient beta")
    axis.set_ylabel("Relative Test L2 (log scale, lower is better)")
    axis.set_title(
        "PG-MS-LR-LPO official eight-beta parameter sweep\n"
        "Three-seed means for FNO/DeepONet; deterministic transport for PG",
        fontsize=14,
        linespacing=1.45,
        pad=16,
    )
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "parameter_sweep_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def milestone_curve(stages: list[tuple[Path, int]]) -> tuple[list[float], list[float]]:
    points = []
    for directory, offset in stages:
        for path in sorted((directory / "milestones").glob("step_*/metrics.json")):
            metrics = read_json(path)
            points.append(
                (
                    (offset + metrics["step"]) / 1000,
                    metrics["test_l2"],
                )
            )
    return [point[0] for point in points], [point[1] for point in points]


def residual_budget() -> None:
    beta2_stages = [
        (RUNS / "beta20" / "pg_residual_sweep_stage1", 0),
        (RUNS / "beta20" / "pg_residual_sweep_stage2", 4000),
        (RUNS / "beta20" / "pg_residual_sweep_stage3", 8000),
        (RUNS / "beta20" / "pg_residual_sweep_stage4", 10000),
        (RUNS / "beta20" / "pg_residual_sweep_stage5", 12000),
        (RUNS / "beta20" / "pg_residual_sweep_stage6", 14000),
        (RUNS / "beta20" / "pg_residual_sweep_stage7", 18000),
        (RUNS / "beta20" / "pg_residual_sweep_stage8", 20000),
        (RUNS / "beta20" / "pg_residual_sweep_stage9", 24000),
        (RUNS / "beta20" / "pg_residual_sweep_stage10", 28000),
        (RUNS / "beta20" / "pg_residual_sweep_stage11", 32000),
    ]
    beta4_stages = [
        (RUNS / "beta40" / "pg_residual_stage1_8k", 0),
        (RUNS / "beta40" / "pg_residual_stage2_12k", 8000),
        (RUNS / "beta40" / "pg_residual_stage3_16k", 12000),
        (RUNS / "beta40" / "pg_residual_stage4_18k", 16000),
    ]
    b2x, b2y = milestone_curve(beta2_stages)
    b4x, b4y = milestone_curve(beta4_stages)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=False)
    for axis in axes:
        style_axis(axis)

    axes[0].plot(b2x, b2y, marker="o", markersize=4, linewidth=2, color=PG)
    axes[0].axhline(0.006389, color=FNO, linestyle="--", linewidth=1.4)
    axes[0].text(1, 0.00645, "FNO", color=FNO, fontsize=9)
    axes[0].scatter([b2x[-1]], [b2y[-1]], s=60, color=PG, zorder=3)
    axes[0].text(
        b2x[-1],
        b2y[-1] * 0.94,
        f"{b2y[-1]:.4f}",
        ha="right",
        color=PG,
        fontsize=9,
        fontweight="bold",
    )
    axes[0].set_xlabel("Residual training step (K)")
    axes[0].set_ylabel("Relative Test L2")
    axes[0].set_title("beta=2 residual budget")

    axes[1].plot(b4x, b4y, marker="o", markersize=5, linewidth=2, color=PG)
    axes[1].axhline(0.007262, color=FNO, linestyle="--", linewidth=1.4)
    axes[1].text(1, 0.00732, "FNO", color=FNO, fontsize=9)
    axes[1].scatter([b4x[-1]], [b4y[-1]], s=60, color=PG, zorder=3)
    axes[1].text(
        b4x[-1],
        b4y[-1] * 0.94,
        f"{b4y[-1]:.4f}",
        ha="right",
        color=PG,
        fontsize=9,
        fontweight="bold",
    )
    axes[1].set_xlabel("Residual training step (K)")
    axes[1].set_title("beta=4 residual budget")

    fig.suptitle(
        "Residual branch convergence and matched-FNO thresholds",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "residual_budget_analysis.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def beta7_ablation() -> None:
    labels = [
        "Fixed transport",
        "64 grid, 16K",
        "128 grid, 8K",
        "+ x_bands 32",
        "+ rollout 17",
        "+ refiner 10p",
        "+ refiner 153p",
        "+ hidden 16",
        "+ hidden 20",
    ]
    values = [
        0.033748,
        0.013577,
        0.013429,
        0.012490,
        0.012297,
        0.011945,
        0.011689,
        0.011494,
        0.011352,
    ]
    colors = [FIXED] + [
        plt.cm.Blues(value)
        for value in np.linspace(0.35, 0.9, len(values) - 1)
    ]
    y = np.arange(len(labels))

    fig, axis = plt.subplots(figsize=(11.5, 6.2))
    style_axis(axis)
    bars = axis.barh(y, values, color=colors)
    axis.axvline(0.009003, color=FNO, linestyle="--", linewidth=1.8)
    axis.text(
        0.00915,
        len(labels) - 0.4,
        "Matched FNO 0.009003",
        color=FNO,
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 0.038)
    axis.set_xlabel("Relative Test L2, lower is better")
    axis.set_title(
        "beta=7 residual branch ablation\n"
        "High-frequency decoder and local correction improve PG, but FNO remains better",
        fontsize=14,
        linespacing=1.4,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "beta7_residual_ablation.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def residual_seed_consistency() -> None:
    summary = read_json(OUT / "residual_seed_summary.json")
    entries = [
        ("beta=2", summary["beta2"]),
        ("beta=4", summary["beta4"]),
        ("beta=7", summary["beta7"]),
    ]
    labels = [entry[0] for entry in entries]
    means = np.array([entry[1]["test_mean"] for entry in entries])
    stds = np.array([entry[1]["test_std"] for entry in entries])
    fno = np.array([entry[1]["fno_test_l2"] for entry in entries])
    x = np.arange(len(entries))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    for axis in axes:
        style_axis(axis)

    axes[0].bar(
        x,
        means,
        width=0.58,
        color=LIGHT,
        edgecolor=PG,
        linewidth=1.3,
        label="Final-stage mean",
    )
    seed_offsets = np.array([-0.12, 0.0, 0.12])
    seed_colors = ["#0b5563", PG, "#54a6ad"]
    for index, (_, entry) in enumerate(entries):
        values = np.asarray(entry["test_values"])
        axes[0].scatter(
            np.full(values.shape, x[index]) + seed_offsets,
            values,
            s=36,
            color=seed_colors,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    axes[0].errorbar(
        x,
        means,
        yerr=stds,
        fmt="D",
        markersize=6,
        color=INK,
        ecolor=INK,
        capsize=5,
        linewidth=1.1,
        zorder=5,
        label="Mean +/- std",
    )
    axes[0].hlines(
        fno,
        x - 0.32,
        x + 0.32,
        colors=FNO,
        linestyles="--",
        linewidth=2,
        label="Matched FNO",
    )
    for index, (mean, std) in enumerate(zip(means, stds)):
        axes[0].text(
            index,
            mean + std + 0.00035,
            f"{mean:.6f}\n+/- {std:.6f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0046, 0.0125)
    axes[0].set_ylabel("Relative Test L2, lower is better")
    axes[0].set_title("Final residual stage: seed spread")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)

    relative_gap = (means / fno - 1.0) * 100.0
    gap_colors = [PG if value < 0 else DEEPONET for value in relative_gap]
    bars = axes[1].barh(
        x,
        relative_gap,
        height=0.54,
        color=gap_colors,
        alpha=0.9,
    )
    axes[1].axvline(0, color=INK, linewidth=1.1)
    for bar, value in zip(bars, relative_gap):
        offset = 1.2 if value >= 0 else -1.2
        alignment = "left" if value >= 0 else "right"
        axes[1].text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            va="center",
            ha=alignment,
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    axes[1].set_yticks(x, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-31, 36)
    axes[1].set_xlabel("Relative difference from matched FNO (negative is better)")
    axes[1].set_title("Accuracy margin against FNO")
    axes[1].text(
        -30,
        2.43,
        "PG better",
        color=PG,
        fontsize=9,
        fontweight="bold",
    )
    axes[1].text(
        35,
        2.43,
        "FNO better",
        color=DEEPONET,
        fontsize=9,
        fontweight="bold",
        ha="right",
    )

    fig.suptitle(
        "Residual branch three-seed stability\n"
        "Shared earlier-stage checkpoint; seeds vary the final fine-tuning stage only",
        fontsize=14,
        fontweight="bold",
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(
        OUT / "residual_seed_summary.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parameter_sweep()
    residual_budget()
    beta7_ablation()
    residual_seed_consistency()


if __name__ == "__main__":
    main()
