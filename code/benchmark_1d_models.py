"""Benchmark 1D PG, FNO, and DeepONet inference cost."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from baselines_1d_burgers import DeepONet, FNO1d, grid_coordinates
from burgers_common import NX, NT, count_parameters
from coord_attention_operator import CoordAttentionOperator, OperatorConfig


def timed_forward(
    forward,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        with torch.no_grad():
            for start, end in zip(starts, ends):
                start.record()
                forward()
                end.record()
        torch.cuda.synchronize(device)
        durations = [
            start.elapsed_time(end) / 1000.0
            for start, end in zip(starts, ends)
        ]
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        for _ in range(warmup):
            forward()
        durations = []
        with torch.no_grad():
            for _ in range(repeats):
                started = time.perf_counter()
                forward()
                durations.append(time.perf_counter() - started)
        peak_memory = 0
    median = statistics.median(durations)
    return {
        "latency_ms": median * 1000.0,
        "latency_p95_ms": sorted(durations)[
            min(len(durations) - 1, int(0.95 * len(durations)))
        ]
        * 1000.0,
        "throughput_samples_per_second": forward.batch_size / median,
        "peak_gpu_memory_mb": peak_memory / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--fno-width", type=int, default=104)
    parser.add_argument("--fno-modes", type=int, default=10)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--deeponet-basis", type=int, default=128)
    parser.add_argument("--deeponet-branch-hidden", type=int, default=512)
    parser.add_argument("--deeponet-trunk-hidden", type=int, default=148)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.pg_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    pg_config = OperatorConfig(
        **{
            key: value
            for key, value in checkpoint["operator"].items()
            if key in OperatorConfig.__dataclass_fields__
        }
    )
    pg = CoordAttentionOperator(pg_config).to(device).eval()
    pg.load_state_dict(checkpoint["state_dict"])
    fno = FNO1d(
        args.fno_modes,
        args.fno_width,
        args.fno_layers,
        NT,
    ).to(device).eval()
    deeponet = DeepONet(
        NX,
        args.deeponet_basis,
        args.deeponet_branch_hidden,
        args.deeponet_trunk_hidden,
    ).to(device).eval()
    initial = torch.randn(args.batch_size, NX, device=device)
    coords = grid_coordinates(NX, NT, device=device)

    models = {
        "PG-MS-LR-LPO": (
            pg,
            lambda: pg.forward_grid(initial),
        ),
        "FNO": (
            fno,
            lambda: fno(initial),
        ),
        "DeepONet-high": (
            deeponet,
            lambda: deeponet(
                initial,
                coords,
            ),
        ),
    }
    results = {}
    for label, (model, forward) in models.items():
        forward.batch_size = args.batch_size
        results[label] = {
            "parameters": count_parameters(model),
            **timed_forward(
                forward,
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
        "batch_size": args.batch_size,
        "grid": [NX, NT],
        "pg_config": asdict(pg_config),
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
