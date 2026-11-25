"""Run matched PINO baselines for Burgers and advection beta=1."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
OUTPUT = ROOT / "experiments" / "pino_formal"

CONFIGS = {
    "burgers": {
        "data_dir": DATASETS / "pdebench_burgers_official",
        "dataset_prefix": "burgers_nu0.01",
        "equation": "burgers",
        "pde_beta": 1.0,
        "width": 104,
        "modes": 10,
        "physics_weight": 1.0e-4,
        "weight_tag": "w0001",
    },
    "advection_beta1": {
        "data_dir": DATASETS / "pdebench_advection_beta1_official",
        "dataset_prefix": "advection_beta1",
        "equation": "advection",
        "pde_beta": 1.0,
        "width": 68,
        "modes": 25,
        "physics_weight": 5.0e-4,
        "weight_tag": "w0005",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(CONFIGS),
        default=list(CONFIGS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--physics-warmup", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for task in args.tasks:
        config = CONFIGS[task]
        for seed in args.seeds:
            suffix = f"{args.steps // 1000}k"
            output = (
                OUTPUT
                / task
                / f"{config['weight_tag']}_s{seed}_{suffix}"
            )
            metrics = output / "metrics.json"
            if metrics.exists() and not args.force:
                print(f"skip {task} seed={seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "code" / "baselines_1d_burgers.py"),
                "--model",
                "pino",
                "--equation",
                config["equation"],
                "--pde-beta",
                str(config["pde_beta"]),
                "--data-dir",
                str(config["data_dir"]),
                "--dataset-prefix",
                config["dataset_prefix"],
                "--train-samples",
                "8000",
                "--dev-samples",
                "1000",
                "--test-samples",
                "1000",
                "--steps",
                str(args.steps),
                "--batch-size",
                "64",
                "--width",
                str(config["width"]),
                "--modes",
                str(config["modes"]),
                "--layers",
                "4",
                "--physics-weight",
                str(config["physics_weight"]),
                "--physics-warmup",
                str(args.physics_warmup),
                "--seed",
                str(seed),
                "--log-every",
                "3000",
                "--save-model",
                "--output-dir",
                str(output),
            ]
            print(f"start {task} seed={seed}", flush=True)
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{task} seed={seed} failed")
            print(f"finish {task} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
