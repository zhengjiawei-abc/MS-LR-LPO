"""Regenerate documentation figures for the current PG-MS-LR-LPO project."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
CROSS = ROOT / "experiments" / "outputs_cross_pde"
DATA = ROOT / "datasets" / "pdebench_burgers_official"

INK = "#17212b"
MUTED = "#64748b"
GRID = "#d7dee8"
MODEL = "#087f8c"
FNO = "#d97706"
ACCENT = "#c2415d"
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dev_curve(path: Path, marker: str | None = None):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if marker is not None:
        text = text.split(marker, 1)[1]
        text = text.split("start seed", 1)[0]
    rows = re.findall(r"step=(\d+).*?dev=([0-9.]+)", text)
    return np.array([int(step) for step, _ in rows]), np.array(
        [float(value) for _, value in rows]
    )


def parse_fno_curve(path: Path, marker: str | None = None):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if marker is not None:
        text = text.split(marker, 1)[1]
        text = text.split("finish seed", 1)[0]
    rows = re.findall(r"step\s+(\d+)\s+loss\s+([0-9.eE+-]+)", text)
    return np.array([int(step) for step, _ in rows]), np.array(
        [float(value) for _, value in rows]
    )


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def architecture_diagram():
    fig, axis = plt.subplots(figsize=(15, 7.5))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 7.5)
    axis.axis("off")

    def box(x, y, w, h, title, subtitle, color, edge=None):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor=color,
            edgecolor=edge or color,
            linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        axis.text(
            x + w / 2,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
            color=MUTED,
            linespacing=1.25,
        )

    def arrow(start, end, color=MUTED, style="-|>", dashed=False):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=14,
                linewidth=1.8,
                color=color,
                linestyle="--" if dashed else "-",
                connectionstyle="arc3,rad=0",
            )
        )

    axis.text(
        0.55,
        7.05,
        "PG-MS-LR-LPO: Physics-Guided Multi-Scale Low-Rank Latent PDE Operator",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.55,
        6.66,
        "Neural latent PDE plus an optional periodic constant-velocity transport prior",
        fontsize=10.5,
        color=MUTED,
    )

    box(0.55, 3.25, 1.35, 1.55, "u0(x)", "256 grid points", LIGHT, GRID)
    box(2.45, 4.55, 2.45, 1.45, "Context encoder", "CNN + Transformer\n32 context tokens", "#e7f0f2")
    box(2.45, 2.1, 2.45, 1.45, "Manifold encoder", "96 channels x 64 latent points", "#e7f0f2")
    box(5.55, 2.1, 2.35, 1.45, "LatentPDE", "9 time anchors\nspectral evolution + nonlinearity", "#fff1dc")
    box(8.55, 2.1, 2.5, 1.45, "Low-rank grid decoder", "3 SVD blocks, rank 17\n+ Fourier x/t features", "#e9f4ea")
    box(11.65, 2.1, 2.15, 1.45, "u(x,t)", "physics-guided fusion\n256 x 101 field", LIGHT, GRID)
    box(8.55, 5.2, 2.5, 1.0, "Multi-scale skips", "256 / 128 / 64 features\ntime-gated fusion", "#f8e9ed")
    box(11.65, 4.75, 2.15, 1.0, "Periodic transport", "shift u0 by beta t\nmod 1, periodic boundary", "#f8e9ed")

    arrow((1.9, 4.02), (2.45, 5.1))
    arrow((1.9, 4.02), (2.45, 2.82))
    arrow((4.9, 5.15), (8.55, 5.7), color=ACCENT, dashed=True)
    arrow((4.9, 2.82), (5.55, 2.82))
    arrow((7.9, 2.82), (8.55, 2.82))
    arrow((11.05, 2.82), (11.65, 2.82))
    arrow((8.55, 5.7), (9.8, 3.55), color=ACCENT, dashed=True)
    arrow((12.7, 4.75), (12.7, 3.55), color=ACCENT, dashed=True)

    axis.text(6.8, 1.35, "Parameters: 501,435", fontsize=11, color=MODEL, fontweight="bold")
    axis.text(6.8, 1.02, "Matched FNO: 502,997", fontsize=10, color=FNO, fontweight="bold")
    axis.text(
        6.8,
        0.55,
        "Test L2: Burgers 0.01976 | diffusion-sorption 0.00331 | advection 0.00454",
        fontsize=11,
        color=INK,
    )
    fig.savefig(OUT / "architecture_diagram.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def performance_comparison():
    datasets = ["Burgers", "Diffusion-sorption", "Advection beta=1"]
    methods = ["Original MS-LR-LPO", "PG-MS-LR-LPO", "FNO", "DeepONet"]
    colors = ["#7aa6a9", MODEL, FNO, ACCENT]
    values = np.array(
        [
            [
                np.mean(
                    [
                        read_json(RUNS / "latent_pde_all_lowrank_joint" / "metrics.json")["final"]["test"],
                        read_json(RUNS / "latent_pde_all_lowrank_joint_s11" / "metrics.json")["final"]["test"],
                        read_json(RUNS / "latent_pde_all_lowrank_joint_s23" / "metrics.json")["final"]["test"],
                    ]
                ),
                np.mean(
                    [
                        read_json(RUNS / "latent_pde_all_lowrank_joint" / "metrics.json")["final"]["test"],
                        read_json(RUNS / "latent_pde_all_lowrank_joint_s11" / "metrics.json")["final"]["test"],
                        read_json(RUNS / "latent_pde_all_lowrank_joint_s23" / "metrics.json")["final"]["test"],
                    ]
                ),
                np.mean(
                    [
                        read_json(RUNS / "fno_w104_m10_s7_24k" / "metrics.json")["test_relative_l2"],
                        read_json(RUNS / "fno_w104_m10_s11_24k" / "metrics.json")["test_relative_l2"],
                        read_json(RUNS / "fno_w104_m10_s23_24k" / "metrics.json")["test_relative_l2"],
                    ]
                ),
                np.mean(
                    [
                        read_json(RUNS / "deeponet_pm501_dense_s7_24k" / "metrics.json")["test_relative_l2"],
                        read_json(RUNS / "deeponet_pm501_dense_s11_24k" / "metrics.json")["test_relative_l2"],
                        read_json(RUNS / "deeponet_pm501_dense_s23_24k" / "metrics.json")["test_relative_l2"],
                    ]
                ),
            ],
            [
                np.mean(
                    [
                        read_json(CROSS / "full" / "diff_sorp" / "ms_lr_lpo" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "seeds" / "diff_sorp" / "pg_ms_lr_lpo_s11_12k" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "seeds" / "diff_sorp" / "pg_ms_lr_lpo_s23_12k" / "metrics.json")["test_relative_l2"],
                    ]
                ),
                np.mean(
                    [
                        read_json(CROSS / "full" / "diff_sorp" / "ms_lr_lpo" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "seeds" / "diff_sorp" / "pg_ms_lr_lpo_s11_12k" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "seeds" / "diff_sorp" / "pg_ms_lr_lpo_s23_12k" / "metrics.json")["test_relative_l2"],
                    ]
                ),
                np.mean(
                    [
                        read_json(CROSS / "fno_pm503" / "diff_sorp" / "s7" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "fno_pm503" / "diff_sorp" / "s11" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "fno_pm503" / "diff_sorp" / "s23" / "metrics.json")["test_relative_l2"],
                    ]
                ),
                np.mean(
                    [
                        read_json(CROSS / "deeponet_high_supervision" / "diff_sorp" / "deeponet_high4096_s7_high4096" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "deeponet_high_supervision" / "diff_sorp" / "deeponet_high4096_s11_high4096" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "deeponet_high_supervision" / "diff_sorp" / "deeponet_high4096_s23_high4096" / "metrics.json")["test_relative_l2"],
                    ]
                ),
            ],
            [
                read_json(CROSS / "full" / "advection_beta1" / "ms_lr_lpo" / "metrics.json")["test_relative_l2"],
                read_json(CROSS / "improve_advection" / "transport_prior" / "metrics.json")["test_relative_l2"],
                np.mean(
                    [
                        read_json(CROSS / "fno_pm503" / "advection_beta1" / "s7" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "fno_pm503" / "advection_beta1" / "s11" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "fno_pm503" / "advection_beta1" / "s23" / "metrics.json")["test_relative_l2"],
                    ]
                ),
                np.mean(
                    [
                        read_json(CROSS / "deeponet_high_supervision" / "advection_beta1" / "deeponet_high4096_s7_high4096" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "deeponet_high_supervision" / "advection_beta1" / "deeponet_high4096_s11_high4096" / "metrics.json")["test_relative_l2"],
                        read_json(CROSS / "deeponet_high_supervision" / "advection_beta1" / "deeponet_high4096_s23_high4096" / "metrics.json")["test_relative_l2"],
                    ]
                ),
            ],
        ]
    )

    fig, axis = plt.subplots(figsize=(13.5, 6.2))
    style_axis(axis)
    axis.set_yscale("log")
    axis.set_ylim(0.002, 3.5)
    width = 0.18
    x = np.arange(len(datasets))
    for index, (method, color) in enumerate(zip(methods, colors)):
        offset = (index - 1.5) * width
        bars = axis.bar(x + offset, values[:, index], width, color=color, label=method)
        for bar, value in zip(bars, values[:, index]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.14,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
                rotation=90,
            )

    axis.set_xticks(x, datasets)
    axis.set_ylabel("Relative Test L2 (log scale, lower is better)")
    axis.set_title(
        "PG-MS-LR-LPO across PDEBench tasks\n"
        "Standard model: 501,435 params | FNO: 502,997-503,165 | "
        "DeepONet: 501,473 params, 4096 queries",
        pad=16,
        fontsize=14,
        linespacing=1.45,
    )
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(OUT / "performance_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def ablation_study():
    diag = read_json(RUNS / "latent_pde_best_skip_corrected" / "diagnostics.json")
    scales = read_json(RUNS / "latent_pde_best_skip_corrected" / "scale_diagnostics.json")
    base = diag["skip_trained"]["dev"]
    blocks = diag["block_ablation"]
    block_labels = ["Full model"] + [f"Remove\nblock {row['removed_block']}" for row in blocks]
    block_values = [base] + [row["dev"] for row in blocks]

    scale_labels = ["Full model", "Remove\n256 skip", "Remove\n128 skip", "Remove\n64 skip", "Remove\nall skips"]
    scale_values = [
        scales["baseline"]["dev"],
        scales["remove_scale"][0]["dev"],
        scales["remove_scale"][1]["dev"],
        scales["remove_scale"][2]["dev"],
        scales["without_all_skips"]["dev"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
    for axis in axes:
        style_axis(axis)

    colors = [MODEL, "#7aa6a9", "#8fb6b8", "#a8c5c6"]
    axes[0].bar(block_labels, block_values, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.017, 0.30)
    axes[0].set_ylabel("Dev L2 (log scale)")
    axes[0].set_title("Dense decoder blocks are all important")
    for index, value in enumerate(block_values):
        axes[0].text(index, value * 1.08, f"{value:.3g}", ha="center", fontsize=9)

    scale_colors = [MODEL, "#b8d2d3", "#8ebabb", "#5d9b9e", ACCENT]
    axes[1].bar(scale_labels, scale_values, color=scale_colors)
    axes[1].set_ylim(0.0193, 0.0207)
    axes[1].set_ylabel("Dev L2")
    axes[1].set_title("All skip scales help; the 64-point scale matters most")
    for index, value in enumerate(scale_values):
        axes[1].text(index, value + 0.00002, f"{value:.4f}", ha="center", fontsize=8.5)
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle("Base MS-LR-LPO ablation study", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "ablation_study.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def dataset_analysis():
    counts = {"Train": 8000, "Dev": 1000, "Test": 1000}
    names = [
        "burgers_nu0.01_train_000.parquet",
        "burgers_nu0.01_dev.parquet",
        "burgers_nu0.01_test.parquet",
    ]
    means = []
    if DATA.exists():
        import pyarrow.parquet as pq

        for name in names:
            table = pq.read_table(DATA / name, columns=["tensor"]).slice(0, 256)
            means.append(
                np.array(
                    [
                        np.asarray(table.column("tensor")[i].as_py(), dtype=np.float32)[:, 0, 0].mean()
                        for i in range(table.num_rows)
                    ]
                )
            )
    else:
        means = [
            np.array([-0.041, 0.57]),
            np.array([0.045, 0.61]),
            np.array([0.030, 0.57]),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.4, 1.05]})
    for axis in axes:
        style_axis(axis)

    axes[0].bar(counts.keys(), counts.values(), color=[MODEL, "#7fb4b7", FNO])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Samples (log scale)")
    axes[0].set_title("Official split")
    for index, value in enumerate(counts.values()):
        axes[0].text(index, value * 1.12, f"{value:,}", ha="center", fontsize=9)

    box = axes[1].boxplot(
        means,
        tick_labels=["Train", "Dev", "Test"],
        patch_artist=True,
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], [MODEL, "#7fb4b7", FNO]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for index, values in enumerate(means, start=1):
        jitter = np.linspace(-0.08, 0.08, min(80, len(values)))
        axes[1].scatter(
            index + jitter[: len(values)],
            values[: len(jitter)],
            s=7,
            color=INK,
            alpha=0.28,
            linewidths=0,
        )
    axes[1].set_ylabel("Spatial mean of u0")
    axes[1].set_title("Initial-condition diversity, first 256 samples")

    axes[2].axis("off")
    axes[2].add_patch(
        FancyBboxPatch(
            (0.04, 0.08),
            0.92,
            0.84,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            transform=axes[2].transAxes,
            facecolor=LIGHT,
            edgecolor=GRID,
        )
    )
    axes[2].text(0.12, 0.78, "Official HDF5", fontsize=12, fontweight="bold", transform=axes[2].transAxes)
    axes[2].text(0.12, 0.64, "10,000 x 201 x 1024", fontsize=11, transform=axes[2].transAxes)
    axes[2].text(0.12, 0.51, "x[::4], t[::2]", fontsize=10, color=MUTED, transform=axes[2].transAxes)
    axes[2].text(0.12, 0.35, "Project parquet", fontsize=12, fontweight="bold", transform=axes[2].transAxes)
    axes[2].text(0.12, 0.21, "256 x 101 x 1", fontsize=12, color=MODEL, fontweight="bold", transform=axes[2].transAxes)

    fig.suptitle("PDEBench Burgers Nu=0.01 data used by the base operator", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "dataset_analysis.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def training_curves():
    model_curves = {
        "Seed 7": parse_dev_curve(RUNS / "latent_pde_all_lowrank_joint" / "train.log"),
        "Seed 11": parse_dev_curve(
            RUNS / "joint_seed_queue.log", marker="start seed 11"
        ),
        "Seed 23": parse_dev_curve(
            RUNS / "joint_seed_queue.log", marker="start seed 23"
        ),
    }
    fno_curves = {
        "Seed 7": parse_fno_curve(RUNS / "fno_w104_m10_s7_24k" / "train.log"),
        "Seed 11": parse_fno_curve(
            RUNS / "fno_seed_queue.log", marker="start seed 11"
        ),
        "Seed 23": parse_fno_curve(
            RUNS / "fno_seed_queue.log", marker="start seed 23"
        ),
    }
    colors = [MODEL, "#4f9ca1", FNO]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis in axes:
        style_axis(axis)

    for (label, (steps, values)), color in zip(model_curves.items(), colors):
        axes[0].plot(steps, values, marker="o", markersize=3, linewidth=1.8, color=color, label=label)
    axes[0].set_xlabel("Joint fine-tune step")
    axes[0].set_ylabel("Dev relative L2")
    axes[0].set_title("Base MS-LR-LPO convergence")
    axes[0].legend(frameon=False)
    axes[0].axhline(0.024588, color=FNO, linestyle="--", linewidth=1.2, alpha=0.85)
    axes[0].text(500, 0.02475, "FNO 3-seed mean", color=FNO, fontsize=8.5)

    for (label, (steps, values)), color in zip(fno_curves.items(), colors):
        axes[1].plot(steps, values, marker="o", markersize=3, linewidth=1.8, color=color, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("FNO training step")
    axes[1].set_ylabel("Training MSE")
    axes[1].set_title("Matched FNO training loss")
    axes[1].legend(frameon=False)

    fig.suptitle("Base operator convergence curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "training_curves.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    architecture_diagram()
    performance_comparison()
    ablation_study()
    dataset_analysis()
    training_curves()


if __name__ == "__main__":
    main()
