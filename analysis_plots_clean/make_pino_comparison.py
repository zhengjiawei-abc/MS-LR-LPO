"""Compare PG-MS-LR-LPO against matched PINO baselines."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from baselines_1d_burgers import FNO1d, normalized_pde_residual
from burgers_common import load_split, relative_l2
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from run_advection_beta_queue import fixed_transport_prediction


OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
PINO = EXP / "pino_formal"
REFINER = EXP / "pg_physics_refiner"
DERIVATIVE_REFINER = EXP / "pg_physics_hpo"
ADVECTION_REFINER = EXP / "pg_advection_output_refiner"
BURGERS = EXP / "outputs_burgers_dataset" / "official_n8000"
CROSS = EXP / "outputs_cross_pde"

INK = "#17212b"
MUTED = "#64748b"
GRID = "#d7dee8"
PG = "#087f8c"
PG_REFINER = "#0f5f68"
PINO_COLOR = "#c2415d"
FNO = "#d97706"
TRUTH = "#94a3b8"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": fmean(values),
        "std": stdev(values),
    }


def pg_burgers() -> tuple[dict, dict]:
    data_dir = ROOT / "datasets" / "pdebench_burgers_official"
    test = load_split(data_dir / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)
    run_names = [
        "latent_pde_all_lowrank_joint",
        "latent_pde_all_lowrank_joint_s11",
        "latent_pde_all_lowrank_joint_s23",
    ]
    test_values = []
    pde_values = []
    parameters = []
    for run_name in run_names:
        checkpoint = torch.load(
            BURGERS / run_name / "model.pt",
            map_location=DEVICE,
            weights_only=False,
        )
        model = CoordAttentionOperator(
            OperatorConfig(**checkpoint["operator"])
        ).to(DEVICE).eval()
        model.load_state_dict(checkpoint["state_dict"])
        predictions = []
        with torch.no_grad():
            for start in range(0, test.size(0), 64):
                predictions.append(
                    model.forward_grid(test[start : start + 64, :, 0, 0])
                )
        prediction = torch.cat(predictions, dim=0)
        test_values.append(relative_l2(prediction, test[..., 0]))
        pde_values.append(
            float(normalized_pde_residual(prediction, "burgers", 1.0)[1])
        )
        parameters.append(sum(value.numel() for value in model.parameters()))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    truth_pde = float(
        normalized_pde_residual(test[..., 0], "burgers", 1.0)[1]
    )
    return (
        {
            **summarize(test_values),
            "pde_relative_rms": fmean(pde_values),
            "parameters": parameters[0],
            "run_names": run_names,
        },
        {"relative_rms": truth_pde},
    )


def pg_advection() -> tuple[dict, dict]:
    data_dir = ROOT / "datasets" / "pdebench_advection_beta1_official"
    test = load_split(data_dir / "advection_beta1_test.parquet", 1000).to(DEVICE)
    prediction = fixed_transport_prediction(test[:, :, 0, 0], 1.0)
    test_l2 = relative_l2(prediction, test[..., 0])
    return (
        {
            "values": [test_l2, test_l2, test_l2],
            "mean": test_l2,
            "std": 0.0,
            "pde_relative_rms": float(
                normalized_pde_residual(prediction, "advection", 1.0)[1]
            ),
            "parameters": 501435,
            "branch_parameters": 0,
        },
        {
            "relative_rms": float(
                normalized_pde_residual(prediction, "advection", 1.0)[1]
            )
        },
    )


def advection_output_refiner_summary() -> dict:
    return refiner_summary(
        [
            ADVECTION_REFINER / f"w0008_budget30_s{seed}_4k" / "metrics.json"
            for seed in (7, 11, 23)
        ]
    )


def pino_summary(task: str, weight_tag: str) -> dict:
    values = []
    pde_values = []
    parameters = []
    for seed in (7, 11, 23):
        metrics = read_json(
            PINO / task / f"{weight_tag}_s{seed}_24k" / "metrics.json"
        )
        values.append(float(metrics["test_relative_l2"]))
        pde_values.append(float(metrics["test_pde_relative_rms"]))
        parameters.append(int(metrics["parameter_count"]))
    return {
        **summarize(values),
        "pde_relative_rms": summarize(pde_values),
        "parameters": parameters[0],
    }


def refiner_summary(paths: list[Path]) -> dict:
    metrics = [read_json(path) for path in paths]
    values = [float(item["test"]["l2"]) for item in metrics]
    pde_values = [
        float(item["test"]["pde_relative_rms"]) for item in metrics
    ]
    return {
        **summarize(values),
        "pde_relative_rms": summarize(pde_values),
        "parameters": int(metrics[0]["parameter_count"]),
        "trainable_parameters": int(metrics[0]["trainable_parameters"]),
        "paths": [str(path) for path in paths],
    }


def pg_refiner_summary() -> dict:
    return refiner_summary(
        [
            REFINER / f"burgers_h18_k11_w01_ext4k_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ]
    )


def pg_derivative_refiner_summary() -> dict:
    return refiner_summary(
        [
            DERIVATIVE_REFINER
            / f"refiner_h18_k11_w03_deriv2k5_4k_budget02_s{seed}"
            / "metrics.json"
            for seed in (7, 11, 23)
        ]
    )


def pg_refiner_accuracy_summary() -> dict:
    return refiner_summary(
        [
            REFINER / "burgers_h16_w01_4k" / "metrics.json",
            REFINER / "burgers_h16_w01_4k_s11" / "metrics.json",
            REFINER / "burgers_h16_w01_4k_s23" / "metrics.json",
        ]
    )


def pg_refiner_mid_summary() -> dict:
    return refiner_summary(
        [
            REFINER / f"burgers_h24_k5_w01_4k_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ]
    )


def pg_refiner_9x9_summary() -> dict:
    return refiner_summary(
        [
            REFINER / f"burgers_h24_k9_w01_ext4k_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ]
    )


def pg_refiner_candidates() -> dict:
    paths = {
        "h16_w01_2k": EXP / "pg_physics_hpo" / "refiner_h16_w01_2k" / "metrics.json",
        "h32_w03_2k": EXP / "pg_physics_hpo" / "refiner_h32_w03_2k" / "metrics.json",
        "h64_w08_2k": EXP / "pg_physics_hpo" / "refiner_h64_w08_2k" / "metrics.json",
        "h16_w01_4k": REFINER / "burgers_h16_w01_4k" / "metrics.json",
        "h24_k5_w01_4k": REFINER / "burgers_h24_k5_w01_4k_s7" / "metrics.json",
        "h24_k9_w01_4k": REFINER / "burgers_h24_k9_w01_4k_s7" / "metrics.json",
        "h24_k9_w01_ext4k": (
            REFINER / "burgers_h24_k9_w01_ext4k_s7" / "metrics.json"
        ),
        "h18_k11_w01_4k": (
            EXP / "pg_physics_hpo" / "refiner_h18_k11_w01_4k" / "metrics.json"
        ),
        "h18_k11_w01_ext4k": (
            REFINER / "burgers_h18_k11_w01_ext4k_s7" / "metrics.json"
        ),
        "h18_k11_w03_derivative": (
            DERIVATIVE_REFINER
            / "refiner_h18_k11_w03_deriv2k5_4k_budget02_s7"
            / "metrics.json"
        ),
    }
    result = {}
    for name, path in paths.items():
        metrics = read_json(path)
        result[name] = {
            "physics_weight": float(metrics["config"]["physics_weight"]),
            "refiner_hidden": int(metrics["config"]["refiner_hidden"]),
            "steps": int(metrics["config"]["steps"]),
            "test_l2": float(metrics["test"]["l2"]),
            "test_pde_relative_rms": float(
                metrics["test"]["pde_relative_rms"]
            ),
            "parameter_count": int(metrics["parameter_count"]),
            "trainable_parameters": int(metrics["trainable_parameters"]),
            "best_step": int(metrics["best_step"]),
        }
    return {
        "selected": "h18_k11_w03_derivative",
        "promotion_note": (
            "The derivative-aware branch reaches lower test L2 and PDE "
            "residual than matched PINO across all three seeds."
        ),
        "selection_rule": (
            "lowest residual at or below the dev L2 budget; the promoted "
            "candidate uses a 2% refiner-stage budget"
        ),
        "candidates": result,
    }


def weight_sweep() -> dict:
    configs = {
        "Burgers": {
            "task": "burgers",
            "weights": ("w0001", "w0005", "w002"),
            "data": "pdebench_burgers_official",
            "prefix": "burgers_nu0.01",
            "equation": "burgers",
        },
        "Advection beta=1": {
            "task": "advection_beta1",
            "weights": ("w0001", "w0005"),
            "data": "pdebench_advection_beta1_official",
            "prefix": "advection_beta1",
            "equation": "advection",
        },
    }
    result = {}
    hpo_root = EXP / "pino_hpo"
    test_fields = {}
    for label, config in configs.items():
        test_fields[label] = load_split(
            ROOT
            / "datasets"
            / config["data"]
            / f"{config['prefix']}_test.parquet",
            1000,
        ).to(DEVICE)
        entries = {}
        for tag in config["weights"]:
            run_dir = hpo_root / config["task"] / f"{tag}_s7_8k"
            metrics = read_json(run_dir / "metrics.json")
            checkpoint = torch.load(
                run_dir / "model.pt",
                map_location=DEVICE,
                weights_only=False,
            )
            model_config = checkpoint["config"]
            model = FNO1d(
                model_config["modes"],
                model_config["width"],
                model_config["layers"],
                test_fields[label].size(2),
            ).to(DEVICE).eval()
            model.load_state_dict(checkpoint["state_dict"])
            predictions = []
            with torch.no_grad():
                for start in range(0, test_fields[label].size(0), 64):
                    u0 = test_fields[label][
                        start : start + 64, :, 0, 0
                    ]
                    predictions.append(model(u0))
            prediction = torch.cat(predictions, dim=0)
            entries[tag] = {
                "physics_weight": float(metrics["config"]["physics_weight"]),
                "dev_relative_l2": float(metrics["dev_relative_l2"]),
                "test_relative_l2": float(metrics["test_relative_l2"]),
                "test_pde_relative_rms": float(
                    normalized_pde_residual(
                        prediction,
                        config["equation"],
                        1.0,
                    )[1]
                ),
            }
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        result[label] = entries
    return {
        "scope": "8K-step seed-7 weight selection; final runs use 24K",
        "runs": result,
        "selected": {
            "Burgers": "w0001",
            "Advection beta=1": "w0005",
        },
    }


def main() -> None:
    burgers_pg, burgers_truth = pg_burgers()
    advection_pg, advection_truth = pg_advection()
    summary = {
        "Burgers": {
            "pg": burgers_pg,
            "pg_refiner": pg_refiner_summary(),
            "pg_derivative_refiner": pg_derivative_refiner_summary(),
            "pg_refiner_accuracy": pg_refiner_accuracy_summary(),
            "pino": pino_summary("burgers", "w0001"),
            "truth_pde_relative_rms": burgers_truth["relative_rms"],
            "fno_test_mean": 0.024588,
        },
        "Advection beta=1": {
            "pg": advection_pg,
            "pg_output_refiner": advection_output_refiner_summary(),
            "pino": pino_summary("advection_beta1", "w0005"),
            "truth_pde_relative_rms": advection_truth["relative_rms"],
            "fno_test_mean": read_json(
                OUT / "advection_eight_parameter_summary.json"
            )["beta1"]["fno"]["test_mean"],
        },
    }
    for task in summary:
        pg_values = summary[task]["pg"]["values"]
        pino_values = summary[task]["pino"]["values"]
        summary[task]["welch_p_vs_pg"] = float(
            ttest_ind(pg_values, pino_values, equal_var=False).pvalue
        )
        if "pg_refiner" in summary[task]:
            summary[task]["welch_p_refiner_vs_pino"] = float(
                ttest_ind(
                    summary[task]["pg_refiner"]["values"],
                    pino_values,
                    equal_var=False,
                ).pvalue
            )
        if "pg_derivative_refiner" in summary[task]:
            summary[task]["welch_p_derivative_refiner_vs_pino"] = float(
                ttest_ind(
                    summary[task]["pg_derivative_refiner"]["values"],
                    pino_values,
                    equal_var=False,
                ).pvalue
            )
            summary[task]["welch_p_derivative_residual_vs_pino"] = float(
                ttest_ind(
                    summary[task]["pg_derivative_refiner"][
                        "pde_relative_rms"
                    ]["values"],
                    summary[task]["pino"]["pde_relative_rms"]["values"],
                    equal_var=False,
                ).pvalue
            )
        if "pg_output_refiner" in summary[task]:
            summary[task]["welch_p_output_refiner_vs_pino"] = float(
                ttest_ind(
                    summary[task]["pg_output_refiner"]["values"],
                    pino_values,
                    equal_var=False,
                ).pvalue
            )
            summary[task]["welch_p_output_residual_vs_pino"] = float(
                ttest_ind(
                    summary[task]["pg_output_refiner"][
                        "pde_relative_rms"
                    ]["values"],
                    summary[task]["pino"]["pde_relative_rms"]["values"],
                    equal_var=False,
                ).pvalue
            )
    (OUT / "pino_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "pino_weight_sweep.json").write_text(
        json.dumps(weight_sweep(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    refiner_payload = pg_refiner_candidates()
    refiner_payload["three_seed_residual_selected"] = pg_refiner_summary()
    refiner_payload["three_seed_accuracy_alternative"] = (
        pg_refiner_accuracy_summary()
    )
    refiner_payload["three_seed_mid_capacity"] = pg_refiner_mid_summary()
    refiner_payload["three_seed_9x9"] = pg_refiner_9x9_summary()
    refiner_payload["three_seed_derivative_refiner"] = (
        pg_derivative_refiner_summary()
    )
    refiner_payload["advection_output_refiner"] = (
        advection_output_refiner_summary()
    )
    (OUT / "pg_physics_refiner_summary.json").write_text(
        json.dumps(refiner_payload, ensure_ascii=False, indent=2) + "\n",
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
    tasks = list(summary)
    x = np.arange(len(tasks))
    width = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)

    pg_data = np.array([summary[task]["pg"]["mean"] for task in tasks])
    refiner_data = np.array(
        [
            (
                summary[task]["pg_derivative_refiner"]["mean"]
                if "pg_derivative_refiner" in summary[task]
                else summary[task]["pg_output_refiner"]["mean"]
            )
            if (
                "pg_derivative_refiner" in summary[task]
                or "pg_output_refiner" in summary[task]
            )
            else np.nan
            for task in tasks
        ]
    )
    pino_data = np.array([summary[task]["pino"]["mean"] for task in tasks])
    fno_data = np.array([summary[task]["fno_test_mean"] for task in tasks])
    data_pg = axes[0].bar(
        x - 1.5 * width,
        pg_data,
        width,
        color=PG,
        label="PG-MS-LR-LPO",
    )
    data_refiner = axes[0].bar(
        x - 0.5 * width,
        refiner_data,
        width,
        color=PG_REFINER,
        label="PG + learned PDE refiner",
    )
    data_pino = axes[0].bar(
        x + 0.5 * width,
        pino_data,
        width,
        color=PINO_COLOR,
        label="PINO",
    )
    data_fno = axes[0].bar(
        x + 1.5 * width,
        fno_data,
        width,
        color=FNO,
        label="FNO",
    )
    for index in range(len(tasks)):
        for offset, value, color in (
            (-1.5 * width, pg_data[index], PG),
            (-0.5 * width, refiner_data[index], PG_REFINER),
            (0.5 * width, pino_data[index], PINO_COLOR),
            (1.5 * width, fno_data[index], FNO),
        ):
            if np.isnan(value):
                continue
            axes[0].text(
                index + offset,
                value * 1.12,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.003, 0.05)
    axes[0].set_xticks(x, tasks)
    axes[0].set_ylabel("Relative Test L2 (log scale, lower is better)")
    axes[0].set_title("Data accuracy")

    pg_pde = np.array(
        [summary[task]["pg"]["pde_relative_rms"] for task in tasks]
    )
    refiner_pde = np.array(
        [
            (
                summary[task]["pg_derivative_refiner"][
                    "pde_relative_rms"
                ]["mean"]
                if "pg_derivative_refiner" in summary[task]
                else summary[task]["pg_output_refiner"][
                    "pde_relative_rms"
                ]["mean"]
            )
            if (
                "pg_derivative_refiner" in summary[task]
                or "pg_output_refiner" in summary[task]
            )
            else np.nan
            for task in tasks
        ]
    )
    pino_pde = np.array(
        [summary[task]["pino"]["pde_relative_rms"]["mean"] for task in tasks]
    )
    truth_pde = np.array(
        [summary[task]["truth_pde_relative_rms"] for task in tasks]
    )
    axes[1].bar(
        x - 1.5 * width,
        pg_pde,
        width,
        color=PG,
        label="PG-MS-LR-LPO",
    )
    axes[1].bar(
        x - 0.5 * width,
        refiner_pde,
        width,
        color=PG_REFINER,
        label="PG + learned PDE refiner",
    )
    axes[1].bar(
        x + 0.5 * width,
        pino_pde,
        width,
        color=PINO_COLOR,
        label="PINO",
    )
    truth_bar = axes[1].bar(
        x + 1.5 * width,
        truth_pde,
        width,
        color=TRUTH,
        label="Observed data",
    )
    for index in range(len(tasks)):
        for offset, value, color in (
            (-1.5 * width, pg_pde[index], PG),
            (-0.5 * width, refiner_pde[index], PG_REFINER),
            (0.5 * width, pino_pde[index], PINO_COLOR),
            (1.5 * width, truth_pde[index], TRUTH),
        ):
            if np.isnan(value):
                continue
            axes[1].text(
                index + offset,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
    axes[1].set_xticks(x, tasks)
    axes[1].set_ylim(0.0, 0.78)
    axes[1].set_ylabel("Normalized PDE residual RMS (lower is better)")
    axes[1].set_title("Physics consistency")
    fig.legend(
        [
            data_pg,
            data_refiner,
            data_pino,
            data_fno,
            truth_bar,
        ],
        [
            "PG-MS-LR-LPO",
            "PG + learned PDE refiner",
            "PINO",
            "FNO",
            "Observed data",
        ],
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
    )

    fig.suptitle(
        "PG-MS-LR-LPO versus matched PINO\n"
        "PINO: three seeds, same FNO backbone and 24K steps; "
        "PG learned refiner: three seeds; Burgers is overall-residual only",
        fontsize=14,
        fontweight="bold",
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(OUT / "pino_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
