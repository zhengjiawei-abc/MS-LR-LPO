"""Run independent joint fine-tunes for seeds 11 and 23."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "code" / "finetune_joint_lowrank.py"
SOURCE = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000" / "latent_pde_all_lowrank" / "model.pt"
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"


for seed in (11, 23):
    output = RUNS / f"latent_pde_all_lowrank_joint_s{seed}"
    if (output / "metrics.json").exists():
        print(f"skip seed {seed}", flush=True)
        continue
    print(f"start seed {seed}", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source", str(SOURCE),
            "--output-dir", str(output),
            "--steps", "12000",
            "--seed", str(seed),
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"seed {seed} failed with code {result.returncode}")
    print(f"finish seed {seed}", flush=True)
