"""Run the two-stage derivative-aware Burgers physics refiner."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REFINER = ROOT / "experiments" / "pg_physics_refiner"
HPO = ROOT / "experiments" / "pg_physics_hpo"
SOURCES = {
    seed: REFINER / f"burgers_h18_k11_w01_ext4k_s{seed}" / "model.pt"
    for seed in (7, 11, 23)
}


def run_stage(
    source: Path,
    output: Path,
    seed: int,
    steps: int,
    lr: float,
    max_l2_increase: float,
    derivative_branch: bool,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "code" / "finetune_pde_residual.py"),
        "--source-checkpoint",
        str(source),
        "--output-dir",
        str(output),
        "--steps",
        str(steps),
        "--batch-size",
        "64",
        "--lr",
        str(lr),
        "--weight-decay",
        "0.0001",
        "--physics-weight",
        "0.03",
        "--physics-warmup",
        "100",
        "--score-weight",
        "0.01",
        "--max-l2-increase",
        str(max_l2_increase),
        "--eval-every",
        "500",
        "--seed",
        str(seed),
        "--equation",
        "burgers",
        "--pde-beta",
        "1.0",
        "--physics-refiner",
        "--refiner-hidden",
        "18",
        "--refiner-kernel",
        "11",
        "--freeze-backbone",
    ]
    if derivative_branch:
        command.extend(
            [
                "--refiner-derivatives",
                "--refiner-derivative-hidden",
                "2",
                "--refiner-derivative-kernel",
                "5",
            ]
        )
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        if seed not in SOURCES:
            raise ValueError(f"unsupported seed: {seed}")
        source = SOURCES[seed]
        if not source.exists():
            raise FileNotFoundError(source)
        warmup = HPO / f"refiner_h18_k11_w03_budget22_4k_s{seed}"
        derivative = (
            HPO / f"refiner_h18_k11_w03_deriv2k5_4k_budget02_s{seed}"
        )
        if args.force or not (warmup / "metrics.json").exists():
            print(f"stage=residual seed={seed}", flush=True)
            run_stage(
                source,
                warmup,
                seed,
                args.steps,
                5.0e-4,
                0.22,
                False,
            )
        else:
            print(f"skip stage=residual seed={seed}", flush=True)
        if args.force or not (derivative / "metrics.json").exists():
            print(f"stage=derivative seed={seed}", flush=True)
            run_stage(
                warmup / "model.pt",
                derivative,
                seed,
                args.steps,
                2.5e-4,
                0.02,
                True,
            )
        else:
            print(f"skip stage=derivative seed={seed}", flush=True)


if __name__ == "__main__":
    main()
