"""Train the output-side PDE refiner for official advection beta=1."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = (
    ROOT
    / "experiments"
    / "outputs_cross_pde"
    / "improve_advection"
    / "transport_prior"
    / "model.pt"
)
DATA = ROOT / "datasets" / "pdebench_advection_beta1_official"
OUTPUT = ROOT / "experiments" / "pg_advection_output_refiner"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--physics-weight", type=float, default=0.008)
    parser.add_argument("--max-l2-increase", type=float, default=0.30)
    parser.add_argument("--tag", default="w0008_budget30")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    for seed in args.seeds:
        run_dir = OUTPUT / f"{args.tag}_s{seed}_4k"
        if (run_dir / "metrics.json").exists() and not args.force:
            print(f"skip seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "code" / "finetune_pde_residual.py"),
            "--source-checkpoint",
            str(SOURCE),
            "--data-dir",
            str(DATA),
            "--dataset-prefix",
            "advection_beta1",
            "--output-dir",
            str(run_dir),
            "--steps",
            str(args.steps),
            "--batch-size",
            "64",
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.0001",
            "--physics-weight",
            str(args.physics_weight),
            "--physics-warmup",
            "100",
            "--score-weight",
            "0.01",
            "--max-l2-increase",
            str(args.max_l2_increase),
            "--eval-every",
            "250",
            "--seed",
            str(seed),
            "--equation",
            "advection",
            "--pde-beta",
            "1.0",
            "--output-refiner",
            "--output-refiner-hidden",
            "8",
            "--output-refiner-kernel",
            "5",
            "--refiner-equation",
            "advection",
            "--refiner-beta",
            "1.0",
            "--freeze-backbone",
        ]
        print(f"start seed={seed}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
