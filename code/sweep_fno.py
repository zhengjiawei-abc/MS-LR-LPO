"""FNO capacity sweep on PDEBench Burgers (one seed).

Configs are ranked by dev relative L2; test L2 is recorded but never used
for selection. Finished runs (metrics.json present) are skipped, so the
sweep can be restarted after an interruption.

    python code/sweep_fno.py                                   # 2375 samples, 50k steps
    python code/sweep_fno.py --train 512 --steps 24000 --grid 16x16 32x8 32x16 64x8 64x16
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

OUT = Path("experiments/outputs_burgers_dataset/fno_sweep")
SEED = 7
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default=2375)
parser.add_argument("--steps", type=int, default=50000)
# width x modes; 32x16 is the original config.
parser.add_argument("--grid", nargs="+", default=["32x16", "64x16", "64x32", "128x16", "128x32"])
cli = parser.parse_args()
TRAIN, STEPS = cli.train, cli.steps
GRID = [tuple(int(v) for v in g.split("x")) for g in cli.grid]


def run_dir(width: int, modes: int) -> Path:
    suffix = "" if STEPS == 50000 else f"_st{STEPS // 1000}k"
    return OUT / f"w{width}_m{modes}_l4_n{TRAIN}_s{SEED}{suffix}"


for width, modes in GRID:
    out = run_dir(width, modes)
    if (out / "metrics.json").exists():
        print(f"skip {out.name}", flush=True)
        continue
    print(f"=== {out.name}", flush=True)
    subprocess.run(
        [sys.executable, "code/baselines_1d_burgers.py", "--model", "fno",
         "--train-samples", str(TRAIN), "--test-samples", "250",
         "--steps", str(STEPS), "--log-every", "5000",
         "--width", str(width), "--modes", str(modes), "--layers", "4",
         "--seed", str(SEED), "--output-dir", str(out)],
        check=True,
    )

rows = []
for width, modes in GRID:
    path = run_dir(width, modes) / "metrics.json"
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append((m["dev_relative_l2"], width, modes, m))
print(f"\n{'width':>5} {'modes':>5} {'params':>9} {'train':>7} {'dev':>7} {'test':>7} {'min':>6}")
for dev, width, modes, m in sorted(rows, key=lambda r: r[0]):
    print(f"{width:>5} {modes:>5} {m['parameter_count']:>9} "
          f"{m['train_relative_l2']:>7.4f} {dev:>7.4f} "
          f"{m['test_relative_l2']:>7.4f} {m['training_seconds'] / 60:>6.1f}")
