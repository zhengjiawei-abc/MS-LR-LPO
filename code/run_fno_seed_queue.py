"""Run matched-capacity FNO seeds 11 and 23."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
DATA = ROOT / "datasets" / "pdebench_burgers_official"

for seed in (11, 23):
    output = RUNS / f"fno_w104_m10_s{seed}_24k"
    if (output / "metrics.json").exists():
        print(f"skip seed {seed}", flush=True)
        continue
    print(f"start seed {seed}", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code" / "baselines_1d_burgers.py"),
            "--model", "fno",
            "--data-dir", str(DATA),
            "--train-samples", "8000",
            "--dev-samples", "1000",
            "--test-samples", "1000",
            "--steps", "24000",
            "--batch-size", "64",
            "--width", "104",
            "--modes", "10",
            "--layers", "4",
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
