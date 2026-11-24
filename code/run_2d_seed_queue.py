"""Run repeatable multi-seed 2D operator and FNO comparisons."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["pg2d", "fno2d"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--divergence-weight", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    trainer = Path(__file__).with_name("train_operator_2d.py")
    runs = []
    for model in args.models:
        for seed in args.seeds:
            output = args.output_root / f"{model}_seed{seed}"
            metrics_path = output / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"skip existing {output}", flush=True)
            else:
                command = [
                    sys.executable,
                    str(trainer),
                    "--model",
                    model,
                    "--data-dir",
                    str(args.data_dir),
                    "--output-dir",
                    str(output),
                    "--steps",
                    str(args.steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(seed),
                    "--divergence-weight",
                    str(args.divergence_weight),
                ]
                print("run " + " ".join(command), flush=True)
                subprocess.run(command, check=True)
            if metrics_path.exists():
                metrics = read_metrics(metrics_path)
                runs.append(
                    {
                        "model": model,
                        "seed": seed,
                        "parameter_count": metrics["parameter_count"],
                        "dev_relative_l2": metrics["dev_relative_l2"],
                        "test_relative_l2": metrics["test_relative_l2"],
                        "training_seconds": metrics["training_seconds"],
                        "stopped_early": metrics["stopped_early"],
                    }
                )

    summary_path = args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"runs": runs}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
