"""Run parameter-matched DeepONet seeds 7, 11, and 23."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
DATA = ROOT / "datasets" / "pdebench_burgers_official"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--query-points", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--output-prefix", default="deeponet_pm501")
    parser.add_argument("--output-suffix", default="24k")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--run-dir", type=Path, default=RUNS)
    parser.add_argument("--dataset-prefix", default="burgers_nu0.01")
    args = parser.parse_args()

    for seed in args.seeds:
        output = (
            args.run_dir
            / f"{args.output_prefix}_s{seed}_{args.output_suffix}"
        )
        if (output / "metrics.json").exists():
            print(f"skip seed {seed}", flush=True)
            continue
        print(f"start seed {seed}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "baselines_1d_burgers.py"),
                "--model",
                "deeponet",
                "--data-dir",
                str(args.data_dir),
                "--dataset-prefix",
                args.dataset_prefix,
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
                "--basis",
                "128",
                "--branch-hidden",
                "512",
                "--trunk-hidden",
                "148",
                "--query-points",
                str(args.query_points),
                "--seed",
                str(seed),
                "--log-every",
                "2000",
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"seed {seed} failed")
        print(f"finish seed {seed}", flush=True)


if __name__ == "__main__":
    main()
