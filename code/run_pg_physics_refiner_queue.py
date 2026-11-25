"""Train the lightweight PG physics refiner for three Burgers seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
OUTPUT = ROOT / "experiments" / "pg_physics_refiner"
SOURCES = {
    7: RUNS / "latent_pde_all_lowrank_joint" / "model.pt",
    11: RUNS / "latent_pde_all_lowrank_joint_s11" / "model.pt",
    23: RUNS / "latent_pde_all_lowrank_joint_s23" / "model.pt",
}
KNOWN_OUTPUTS = {
    ("burgers_h16_w01_4k", 7): OUTPUT / "burgers_h16_w01_4k",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--physics-weight", type=float, default=0.01)
    parser.add_argument("--refiner-hidden", type=int, default=16)
    parser.add_argument("--refiner-kernel", type=int, default=3)
    parser.add_argument("--tag", default="burgers_h16_w01_4k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        if seed not in SOURCES:
            raise ValueError(f"unsupported seed: {seed}")
        output = KNOWN_OUTPUTS.get(
            (args.tag, seed),
            OUTPUT / f"{args.tag}_s{seed}",
        )
        if (output / "metrics.json").exists() and not args.force:
            print(f"skip seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "code" / "finetune_pde_residual.py"),
            "--source-checkpoint",
            str(SOURCES[seed]),
            "--output-dir",
            str(output),
            "--steps",
            str(args.steps),
            "--eval-every",
            "500",
            "--physics-weight",
            str(args.physics_weight),
            "--physics-warmup",
            "200",
            "--lr",
            "0.001",
            "--physics-refiner",
            "--refiner-hidden",
            str(args.refiner_hidden),
            "--refiner-kernel",
            str(args.refiner_kernel),
            "--freeze-backbone",
            "--max-l2-increase",
            "0.05",
            "--seed",
            str(seed),
        ]
        print(f"start seed={seed}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"seed={seed} failed")
        print(f"finish seed={seed}", flush=True)


if __name__ == "__main__":
    main()
