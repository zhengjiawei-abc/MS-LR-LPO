"""Run MS-LR-LPO, PG-MS-LR-LPO, parameter-matched FNO, and DeepONet."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "experiments" / "outputs_cross_pde"
DATASETS = {
    "advection_beta1": ROOT / "datasets" / "pdebench_advection_beta1_official",
    "diff_sorp": ROOT / "datasets" / "pdebench_diff_sorp_official",
}


def current_command(dataset_dir: Path, prefix: str, output: Path, args) -> list[str]:
    train_samples = "16" if args.smoke else "8000"
    batch_size = "4" if args.smoke else "64"
    eval_every = "2" if args.smoke else "1000"
    return [
        sys.executable,
        str(ROOT / "code" / "train_coord_attention.py"),
        "--data-dir", str(dataset_dir),
        "--dataset-prefix", prefix,
        "--train-samples", train_samples,
        "--steps", str(args.current_steps),
        "--batch-size", batch_size,
        "--loss-mode", "full",
        "--output-mode", "grid",
        "--grid-hidden", "48",
        "--grid-blocks", "3",
        "--grid-length", "64",
        "--grid-lowrank-rank", "17",
        "--dynamics", "latent_pde",
        "--latent-channels", "96",
        "--latent-length", "64",
        "--pde-layers", "1",
        "--pde-rank", "24",
        "--pde-modes", "32",
        "--rollout-steps", "9",
        "--eval-every", eval_every,
        "--seed", str(args.seed),
        "--save-model",
        "--output-dir", str(output),
    ]


def fno_command(dataset_dir: Path, prefix: str, output: Path, args) -> list[str]:
    samples = ("16", "8", "8") if args.smoke else ("8000", "1000", "1000")
    batch_size = "4" if args.smoke else "64"
    log_every = "2" if args.smoke else "2000"
    return [
        sys.executable,
        str(ROOT / "code" / "baselines_1d_burgers.py"),
        "--model", "fno",
        "--data-dir", str(dataset_dir),
        "--dataset-prefix", prefix,
        "--train-samples", samples[0],
        "--dev-samples", samples[1],
        "--test-samples", samples[2],
        "--steps", str(args.baseline_steps),
        "--batch-size", batch_size,
        "--width", "104",
        "--modes", "10",
        "--layers", "4",
        "--seed", str(args.seed),
        "--log-every", log_every,
        "--output-dir", str(output),
    ]


def physics_command(
    dataset_dir: Path,
    prefix: str,
    base_checkpoint: Path,
    output: Path,
    args,
) -> list[str]:
    velocity = "1.0" if prefix == "advection_beta1" else "0.0"
    command = [
        sys.executable,
        str(ROOT / "code" / "train_coord_attention.py"),
        "--data-dir", str(dataset_dir),
        "--dataset-prefix", prefix,
        "--grid-transport-velocity", velocity,
        "--grid-transport-blend", "0.0",
        "--init-checkpoint", str(base_checkpoint),
        "--evaluate-only",
        "--save-model",
        "--output-dir", str(output),
    ]
    if args.smoke:
        command.extend(
            [
                "--train-samples", "16",
                "--dev-samples", "8",
                "--test-samples", "8",
            ]
        )
    return command


def deeponet_command(
    dataset_dir: Path,
    prefix: str,
    output: Path,
    args,
) -> list[str]:
    samples = ("16", "8", "8") if args.smoke else ("8000", "1000", "1000")
    batch_size = "4" if args.smoke else "64"
    log_every = "2" if args.smoke else "2000"
    return [
        sys.executable,
        str(ROOT / "code" / "baselines_1d_burgers.py"),
        "--model", "deeponet",
        "--data-dir", str(dataset_dir),
        "--dataset-prefix", prefix,
        "--train-samples", samples[0],
        "--dev-samples", samples[1],
        "--test-samples", samples[2],
        "--steps", str(args.baseline_steps),
        "--batch-size", batch_size,
        "--basis", "128",
        "--branch-hidden", "512",
        "--trunk-hidden", "148",
        "--query-points", "256",
        "--seed", str(args.seed),
        "--log-every", log_every,
        "--output-dir", str(output),
    ]


def run_job(name: str, command: list[str], output: Path, force: bool) -> None:
    metrics = output / "metrics.json"
    if metrics.exists() and not force:
        print(f"skip {name}: {metrics}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    print(f"start {name}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
    print(f"finish {name} in {time.time() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--current-steps", type=int, default=12000)
    parser.add_argument("--baseline-steps", type=int, default=24000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    suffix = "smoke" if args.smoke else "full"
    if args.smoke:
        args.current_steps = 4
        args.baseline_steps = 4

    summary = []
    for dataset, data_dir in DATASETS.items():
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        prefix = dataset
        jobs = [
            (
                "ms_lr_lpo",
                current_command(
                    data_dir,
                    prefix,
                    OUTPUT / suffix / dataset / "ms_lr_lpo",
                    args,
                ),
            ),
            (
                "pg_ms_lr_lpo",
                physics_command(
                    data_dir,
                    prefix,
                    OUTPUT / suffix / dataset / "ms_lr_lpo" / "model.pt",
                    OUTPUT / suffix / dataset / "pg_ms_lr_lpo",
                    args,
                ),
            ),
            (
                "fno",
                fno_command(
                    data_dir,
                    prefix,
                    OUTPUT / suffix / dataset / "fno",
                    args,
                ),
            ),
            (
                "deeponet",
                deeponet_command(
                    data_dir,
                    prefix,
                    OUTPUT / suffix / dataset / "deeponet",
                    args,
                ),
            ),
        ]
        for model, command in jobs:
            output = OUTPUT / suffix / dataset / model
            run_job(f"{suffix}/{dataset}/{model}", command, output, args.force)
            metrics_path = output / "metrics.json"
            summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                }
            )
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / f"matrix_{suffix}.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
