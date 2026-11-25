"""Run final residual-branch seed variants for advection beta 2/4/7."""
from __future__ import annotations

import argparse
import json
from statistics import fmean, stdev
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "outputs_cross_pde" / "advection_parameters"

CONFIGS = {
    2.0: {
        "beta_tag": "beta20",
        "prefix": "advection_beta20",
        "steps": 4000,
        "init": "pg_residual_sweep_stage10/model.pt",
        "seed7_output": "pg_residual_sweep_stage10",
        "fno_test_l2": 0.006389,
        "grid_length": 64,
        "x_bands": 16,
        "rollout_steps": 9,
        "refine": False,
        "refine_hidden": 0,
    },
    4.0: {
        "beta_tag": "beta40",
        "prefix": "advection_beta40",
        "steps": 2000,
        "init": "pg_residual_stage3_16k/model.pt",
        "seed7_output": "pg_residual_stage4_18k",
        "fno_test_l2": 0.007262,
        "grid_length": 64,
        "x_bands": 16,
        "rollout_steps": 9,
        "refine": False,
        "refine_hidden": 0,
    },
    7.0: {
        "beta_tag": "beta70",
        "prefix": "advection_beta70",
        "steps": 2000,
        "init": (
            "pg_residual_grid128_xbands30_roll17_refine16_2k/model.pt"
        ),
        "seed7_output": "pg_residual_grid128_xbands30_roll17_refine20_2k",
        "fno_test_l2": 0.009003,
        "grid_length": 128,
        "x_bands": 30,
        "rollout_steps": 17,
        "refine": True,
        "refine_hidden": 20,
    },
}


def write_summary(betas: list[float]) -> Path:
    summary = {}
    for beta in betas:
        config = CONFIGS[beta]
        run_dir = RUNS / config["beta_tag"]
        metric_paths = {
            7: run_dir / config["seed7_output"] / "metrics.json",
            11: run_dir / "pg_residual_seed11_final" / "metrics.json",
            23: run_dir / "pg_residual_seed23_final" / "metrics.json",
        }
        missing = [path for path in metric_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "missing metrics for beta "
                f"{beta:g}: " + ", ".join(str(path) for path in missing)
            )
        metrics = {
            seed: json.loads(path.read_text(encoding="utf-8"))
            for seed, path in metric_paths.items()
        }
        values = [
            float(metrics[seed]["test_relative_l2"])
            for seed in (7, 11, 23)
        ]
        parameter_counts = {int(item["parameter_count"]) for item in metrics.values()}
        if len(parameter_counts) != 1:
            raise ValueError(f"inconsistent parameter counts for beta {beta:g}")
        summary[f"beta{int(beta)}"] = {
            "seeds": [7, 11, 23],
            "test_values": values,
            "test_mean": fmean(values),
            "test_std": stdev(values),
            "parameters": parameter_counts.pop(),
            "fno_test_l2": config["fno_test_l2"],
            "initial_checkpoint": config["init"],
            "final_stage_steps": config["steps"],
            "scope": (
                "final residual stage; shared earlier-stage checkpoint, "
                "seeds vary final fine-tuning only"
            ),
        }
    output = ROOT / "analysis_plots_clean" / "residual_seed_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", nargs="+", type=float, default=[2.0, 4.0, 7.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23])
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        write_summary(args.betas)
        return

    for beta in args.betas:
        if beta not in CONFIGS:
            raise ValueError(f"unsupported beta: {beta}")
        config = CONFIGS[beta]
        data_dir = ROOT / "datasets" / f"pdebench_{config['prefix']}_official"
        run_dir = RUNS / config["beta_tag"]
        init_checkpoint = run_dir / config["init"]
        if not init_checkpoint.exists():
            raise FileNotFoundError(init_checkpoint)
        for seed in args.seeds:
            output = run_dir / f"pg_residual_seed{seed}_final"
            metrics = output / "metrics.json"
            if metrics.exists():
                print(f"skip beta={beta} seed={seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "code" / "train_coord_attention.py"),
                "--data-dir",
                str(data_dir),
                "--dataset-prefix",
                config["prefix"],
                "--train-samples",
                "8000",
                "--steps",
                str(config["steps"]),
                "--batch-size",
                "64",
                "--loss-mode",
                "full",
                "--output-mode",
                "grid",
                "--grid-hidden",
                "48",
                "--grid-blocks",
                "3",
                "--grid-length",
                str(config["grid_length"]),
                "--grid-lowrank-rank",
                "17",
                "--dynamics",
                "latent_pde",
                "--latent-channels",
                "96",
                "--latent-length",
                "64",
                "--pde-layers",
                "1",
                "--pde-rank",
                "24",
                "--pde-modes",
                "32",
                "--pde-substeps",
                "2",
                "--rollout-steps",
                str(config["rollout_steps"]),
                "--x-bands",
                str(config["x_bands"]),
                "--grid-transport-velocity",
                str(beta),
                "--grid-transport-blend",
                "0.15",
                "--grid-transport-learnable",
                "true",
                "--lr",
                "2.5e-5",
                "--warmup",
                "50",
                "--eval-every",
                "500",
                "--save-every",
                "1000",
                "--test-every",
                "1000",
                "--seed",
                str(seed),
                "--init-checkpoint",
                str(init_checkpoint),
                "--save-model",
                "--output-dir",
                str(output),
            ]
            if config["refine"]:
                command.extend(
                    [
                        "--grid-transport-refine",
                        "true",
                        "--grid-transport-refine-hidden",
                        str(config["refine_hidden"]),
                    ]
                )
            print("run " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    write_summary(args.betas)


if __name__ == "__main__":
    main()
