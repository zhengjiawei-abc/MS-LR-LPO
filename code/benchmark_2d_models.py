"""Benchmark parameter count, inference latency, throughput, and memory."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from analyze_2d_models import load_model
from operator_2d_common import load_npz_split, load_spec


def benchmark_model(
    model,
    inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_grid(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for start, end in zip(starts, ends):
                start.record()
                model.forward_grid(inputs)
                end.record()
            torch.cuda.synchronize(device)
            durations = [
                start.elapsed_time(end) / 1000.0
                for start, end in zip(starts, ends)
            ]
        else:
            durations = []
            for _ in range(repeats):
                started = time.perf_counter()
                model.forward_grid(inputs)
                durations.append(time.perf_counter() - started)
    median_seconds = statistics.median(durations)
    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return {
        "seconds_per_batch": median_seconds,
        "latency_ms": median_seconds * 1000.0,
        "latency_p95_ms": sorted(durations)[
            min(len(durations) - 1, int(0.95 * len(durations)))
        ]
        * 1000.0,
        "throughput_samples_per_second": inputs.size(0) / median_seconds,
        "peak_gpu_memory_mb": peak_memory / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pg-checkpoint", type=Path, required=True)
    parser.add_argument("--fno-checkpoint", type=Path, required=True)
    parser.add_argument("--deeponet-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_spec(args.data_dir / "metadata.json")
    inputs, _ = load_npz_split(args.data_dir / "test.npz")
    inputs = inputs[: args.batch_size].to(device)
    checkpoints = [
        ("PG-MS-LR-LPO-2D", args.pg_checkpoint),
        ("FNO-2D", args.fno_checkpoint),
    ]
    if args.deeponet_checkpoint is not None:
        checkpoints.append(("DeepONet-2D", args.deeponet_checkpoint))

    results = {}
    for label, checkpoint in checkpoints:
        model, _ = load_model(checkpoint, spec, device)
        parameters = sum(value.numel() for value in model.parameters())
        metrics_path = checkpoint.parent / "metrics.json"
        training_seconds = None
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            training_seconds = metrics.get("training_seconds")
        results[label] = {
            "parameters": parameters,
            "checkpoint_mb": checkpoint.stat().st_size / (1024**2),
            "training_seconds": training_seconds,
            **benchmark_model(
                model,
                inputs,
                device,
                args.warmup,
                args.repeats,
            ),
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "cpu",
        "batch_size": inputs.size(0),
        "grid": [spec.nx, spec.ny, spec.nt],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
