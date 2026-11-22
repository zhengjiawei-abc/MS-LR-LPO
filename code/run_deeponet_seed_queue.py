"""Run parameter-matched DeepONet seeds 7, 11, and 23."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
DATA = ROOT / "datasets" / "pdebench_burgers_official"

for seed in (7, 11, 23):
    output = RUNS / f"deeponet_pm501_s{seed}_24k"
    if (output / "metrics.json").exists():
        print(f"skip seed {seed}", flush=True)
        continue
    print(f"start seed {seed}", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "baselines_1d_burgers.py"),
            "--model", "deeponet",
            "--data-dir", str(DATA),
            "--train-samples", "8000",
            "--dev-samples", "1000",
            "--test-samples", "1000",
            "--steps", "24000",
            "--batch-size", "64",
            "--basis", "128",
            "--branch-hidden", "512",
            "--trunk-hidden", "148",
            "--query-points", "256",
            "--seed", str(seed),
            "--log-every", "2000",
            "--output-dir", str(output),
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"seed {seed} failed")
    print(f"finish seed {seed}", flush=True)
