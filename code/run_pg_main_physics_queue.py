"""Run late main-training PDE residual continuation for Burgers."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
BASE_RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
OUTPUT = ROOT / "experiments" / "pg_main_physics_late"
SOURCES = {
    7: BASE_RUNS / "latent_pde_all_lowrank_joint" / "model.pt",
    11: BASE_RUNS / "latent_pde_all_lowrank_joint_s11" / "model.pt",
    23: BASE_RUNS / "latent_pde_all_lowrank_joint_s23" / "model.pt",
}
KNOWN_OUTPUTS = {
    7: OUTPUT / "burgers_w0001_s7_6k_from_base_rank16",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--physics-weight", type=float, default=1.0e-4)
    parser.add_argument("--physics-warmup", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        if seed not in SOURCES:
            raise ValueError(f"unsupported seed: {seed}")
        source = SOURCES[seed]
        if not source.exists():
            raise FileNotFoundError(source)
        output = KNOWN_OUTPUTS.get(
            seed,
            OUTPUT / f"burgers_w0001_s{seed}_6k_from_base_rank16",
        )
        if (output / "metrics.json").exists() and not args.force:
            print(f"skip seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "code" / "train_coord_attention.py"),
            "--data-dir",
            str(DATA),
            "--dataset-prefix",
            "burgers_nu0.01",
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
            "--loss-mode",
            "full",
            "--output-mode",
            "grid",
            "--grid-hidden",
            "48",
            "--grid-blocks",
            "3",
            "--grid-length",
            "64",
            "--grid-lowrank-rank",
            "16",
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
            "1",
            "--rollout-steps",
            "9",
            "--physics-weight",
            str(args.physics_weight),
            "--physics-equation",
            "burgers",
            "--physics-start-step",
            "0",
            "--physics-warmup",
            str(args.physics_warmup),
            "--physics-score-weight",
            "0.0",
            "--lr",
            str(args.lr),
            "--warmup",
            "200",
            "--eval-every",
            "500",
            "--save-every",
            "2000",
            "--test-every",
            "2000",
            "--seed",
            str(seed),
            "--init-checkpoint",
            str(source),
            "--save-model",
            "--output-dir",
            str(output),
        ]
        print(f"start seed={seed}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"seed={seed} failed")
        print(f"finish seed={seed}", flush=True)


if __name__ == "__main__":
    main()
