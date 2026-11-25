"""Evaluate fixed transport and matched FNO/DeepONet for new advection betas."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean, stdev

import torch

from burgers_common import NT, NX, T_FINAL, load_split, relative_l2


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "outputs_cross_pde" / "advection_parameters"
SUMMARY = ROOT / "analysis_plots_clean" / "advection_new_beta_summary.json"

CONFIGS = {
    0.1: {
        "run_tag": "advection_beta01",
        "prefix": "advection_beta01",
        "data_dir": "pdebench_advection_beta01_official",
    },
    0.4: {
        "run_tag": "advection_beta04",
        "prefix": "advection_beta04",
        "data_dir": "pdebench_advection_beta04_official",
    },
}


def fixed_transport_prediction(u0: torch.Tensor, beta: float) -> torch.Tensor:
    x = torch.arange(NX, dtype=torch.float32)
    t = torch.linspace(0.0, T_FINAL, NT)
    position = (x[:, None] - beta * t[None, :] * NX).remainder(NX)
    x0 = torch.floor(position).long()
    x1 = (x0 + 1).remainder(NX)
    fraction = position - x0.float()
    source = u0.unsqueeze(-1).expand(-1, -1, NT)
    index0 = x0.unsqueeze(0).expand(u0.size(0), -1, -1)
    index1 = x1.unsqueeze(0).expand(u0.size(0), -1, -1)
    lower = torch.gather(source, 1, index0)
    upper = torch.gather(source, 1, index1)
    return lower + fraction * (upper - lower)


def evaluate_fixed(beta: float) -> Path:
    config = CONFIGS[beta]
    data_dir = ROOT / "datasets" / config["data_dir"]
    output = RUNS / config["run_tag"] / "pg_fixed"
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        print(f"skip fixed beta={beta:g}", flush=True)
        return metrics_path
    target = load_split(
        data_dir / f"{config['prefix']}_test.parquet",
        1000,
    )[..., 0]
    prediction = fixed_transport_prediction(target[:, :, 0], beta)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "analytic_periodic_transport",
        "beta": beta,
        "method": "linear periodic interpolation",
        "branch_parameters": 0,
        "parameter_budget": 501435,
        "test_samples": target.size(0),
        "test_relative_l2": relative_l2(prediction, target),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"fixed beta={beta:g}: {metrics['test_relative_l2']:.8f}", flush=True)
    return metrics_path


def run_command(
    name: str,
    command: list[str],
    output: Path,
    force: bool,
) -> Path:
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"skip {name}", flush=True)
        return metrics_path
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    print(f"start {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed; see {log_path}")
    print(f"finish {name}", flush=True)
    return metrics_path


def baseline_command(
    model: str,
    beta: float,
    seed: int,
    output: Path,
    steps: int,
) -> list[str]:
    config = CONFIGS[beta]
    command = [
        sys.executable,
        str(ROOT / "code" / "baselines_1d_burgers.py"),
        "--model",
        model,
        "--data-dir",
        str(ROOT / "datasets" / config["data_dir"]),
        "--dataset-prefix",
        config["prefix"],
        "--train-samples",
        "8000",
        "--dev-samples",
        "1000",
        "--test-samples",
        "1000",
        "--steps",
        str(steps),
        "--batch-size",
        "64",
        "--seed",
        str(seed),
        "--log-every",
        "2000",
        "--output-dir",
        str(output),
    ]
    if model == "fno":
        command.extend(["--width", "68", "--modes", "25", "--layers", "4"])
    else:
        command.extend(
            [
                "--basis",
                "128",
                "--branch-hidden",
                "512",
                "--trunk-hidden",
                "148",
                "--query-points",
                "256",
            ]
        )
    return command


def read_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(betas: list[float]) -> Path:
    summary = {}
    for beta in betas:
        config = CONFIGS[beta]
        run_dir = RUNS / config["run_tag"]
        fixed_path = run_dir / "pg_fixed" / "metrics.json"
        if not fixed_path.exists():
            evaluate_fixed(beta)
        entry = {
            "fixed_transport": read_metric(fixed_path)["test_relative_l2"],
            "branch_parameters": 0,
            "parameter_budget": 501435,
        }
        for model, prefix in (("fno", "fno_m25_w68"), ("deeponet", "deeponet")):
            values = []
            parameters = []
            for seed in (7, 11, 23):
                path = run_dir / f"{prefix}_s{seed}" / "metrics.json"
                if not path.exists():
                    continue
                metrics = read_metric(path)
                values.append(float(metrics["test_relative_l2"]))
                parameters.append(int(metrics["parameter_count"]))
            if values:
                entry[model] = {
                    "seeds": [7, 11, 23][: len(values)],
                    "test_values": values,
                    "test_mean": fmean(values),
                    "test_std": stdev(values) if len(values) > 1 else 0.0,
                    "parameters": parameters[0],
                }
        summary[f"beta{beta:g}"] = entry
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SUMMARY}", flush=True)
    return SUMMARY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", nargs="+", type=float, default=[0.1, 0.4])
    parser.add_argument("--models", nargs="+", choices=("fixed", "fno", "deeponet"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        write_summary(args.betas)
        return
    models = args.models or ["fixed", "fno", "deeponet"]
    for beta in args.betas:
        if beta not in CONFIGS:
            raise ValueError(f"unsupported beta: {beta}")
        if "fixed" in models:
            evaluate_fixed(beta)
        for model in ("fno", "deeponet"):
            if model not in models:
                continue
            prefix = "fno_m25_w68" if model == "fno" else "deeponet"
            for seed in args.seeds:
                output = RUNS / CONFIGS[beta]["run_tag"] / f"{prefix}_s{seed}"
                run_command(
                    f"beta={beta:g}/{model}/seed={seed}",
                    baseline_command(model, beta, seed, output, args.steps),
                    output,
                    args.force,
                )
    write_summary(args.betas)


if __name__ == "__main__":
    main()
