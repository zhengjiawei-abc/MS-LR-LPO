"""Run latent PDE, coordinate baseline, and FNO on official PDEBench data."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
OUTPUT = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
STEPS = 24000
SEED = 7

RUNS = [
    (
        "latent_pde_s7_24k",
        [
            sys.executable,
            str(ROOT / "code" / "train_coord_attention.py"),
            "--train-samples", "8000",
            "--data-dir", str(DATA),
            "--steps", str(STEPS),
            "--batch-size", "32",
            "--points", "256",
            "--eval-every", "1000",
            "--seed", str(SEED),
            "--dynamics", "latent_pde",
            "--output-dir", str(OUTPUT / "latent_pde_s7_24k"),
            "--save-model",
        ],
    ),
    (
        "coord_baseline_s7_24k",
        [
            sys.executable,
            str(ROOT / "code" / "train_coord_attention.py"),
            "--train-samples", "8000",
            "--data-dir", str(DATA),
            "--steps", str(STEPS),
            "--batch-size", "32",
            "--points", "256",
            "--eval-every", "1000",
            "--seed", str(SEED),
            "--dynamics", "none",
            "--output-dir", str(OUTPUT / "coord_baseline_s7_24k"),
            "--save-model",
        ],
    ),
    (
        "fno_w64_m16_s7_24k",
        [
            sys.executable,
            str(ROOT / "code" / "baselines_1d_burgers.py"),
            "--model", "fno",
            "--data-dir", str(DATA),
            "--train-samples", "8000",
            "--dev-samples", "1000",
            "--test-samples", "1000",
            "--steps", str(STEPS),
            "--batch-size", "64",
            "--width", "64",
            "--modes", "16",
            "--layers", "4",
            "--seed", str(SEED),
            "--log-every", "2000",
            "--output-dir", str(OUTPUT / "fno_w64_m16_s7_24k"),
        ],
    ),
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, command in RUNS:
        directory = OUTPUT / name
        metrics = directory / "metrics.json"
        if metrics.exists():
            print(f"skip {name}", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "train.log"
        print(f"start {name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed with code {result.returncode}")
        print(f"finish {name}", flush=True)


if __name__ == "__main__":
    main()
