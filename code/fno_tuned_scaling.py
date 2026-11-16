"""Tuned FNO (width 64, modes 16, 4 layers, 50k steps; chosen by dev L2 in
sweep_fno.py) at 512 / 1024 / 2375 train samples, seeds 7 / 11 / 23.

Finished runs are skipped. Prints mean ± std of dev and test L2 per size.

    python code/fno_tuned_scaling.py
    python code/fno_tuned_scaling.py --sizes 512 --width 64 --modes 8 --steps 24000
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

OUT = Path("experiments/outputs_burgers_dataset/fno_tuned_scaling")
SEEDS, LAYERS = (7, 11, 23), 4
parser = argparse.ArgumentParser()
parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2375])
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--modes", type=int, default=16)
parser.add_argument("--steps", type=int, default=50000)
cli = parser.parse_args()
SIZES, WIDTH, MODES, STEPS = cli.sizes, cli.width, cli.modes, cli.steps


def run_dir(n: int, seed: int) -> Path:
    suffix = "" if STEPS == 50000 else f"_st{STEPS // 1000}k"
    return OUT / f"fno_w{WIDTH}_m{MODES}_n{n}_s{seed}{suffix}"


for n in SIZES:
    for seed in SEEDS:
        out = run_dir(n, seed)
        if (out / "metrics.json").exists():
            print(f"skip {out.name}", flush=True)
            continue
        print(f"=== {out.name}", flush=True)
        subprocess.run(
            [sys.executable, "code/baselines_1d_burgers.py", "--model", "fno",
             "--train-samples", str(n), "--test-samples", "250",
             "--steps", str(STEPS), "--log-every", "5000",
             "--width", str(WIDTH), "--modes", str(MODES),
             "--layers", str(LAYERS), "--seed", str(seed),
             "--output-dir", str(out)],
            check=True,
        )

summary = {}
print(f"\n{'n':>5} {'dev':>18} {'test':>18}")
for n in SIZES:
    ms = [json.loads((run_dir(n, s) / "metrics.json").read_text(encoding="utf-8"))
          for s in SEEDS]
    dev = [m["dev_relative_l2"] for m in ms]
    test = [m["test_relative_l2"] for m in ms]
    summary[run_dir(n, SEEDS[0]).name.replace(f"_s{SEEDS[0]}", "")] = {
        str(s): t for s, t in zip(SEEDS, test)}
    print(f"{n:>5} {st.mean(dev):.4f} ± {st.stdev(dev):.4f}"
          f" {st.mean(test):.4f} ± {st.stdev(test):.4f}")
path = OUT / "summary.json"
old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
old = {k: v for k, v in old.items() if not k.isdigit()}  # drop pre-CLI keys
path.write_text(json.dumps({**old, **summary}, indent=2), encoding="utf-8")
