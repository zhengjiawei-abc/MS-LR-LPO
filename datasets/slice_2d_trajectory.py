"""Slice processed 2D trajectories to a later identifiable initial state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-start", type=int, required=True)
    args = parser.parse_args()

    metadata = json.loads(
        (args.input_dir / "metadata.json").read_text(encoding="utf-8")
    )
    original_nt = int(metadata["nt"])
    if not 0 <= args.time_start < original_nt - 1:
        raise ValueError(
            f"time start {args.time_start} is invalid for nt={original_nt}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        with np.load(args.input_dir / f"{split}.npz") as source:
            targets = np.asarray(source["targets"])
        sliced = targets[..., args.time_start :]
        np.savez(
            args.output_dir / f"{split}.npz",
            inputs=sliced[..., 0],
            targets=sliced,
        )
        print(
            f"{split}: {targets.shape} -> {sliced.shape}",
            flush=True,
        )

    t_min = float(metadata["t_min"])
    t_max = float(metadata["t_max"])
    dt = (t_max - t_min) / max(original_nt - 1, 1)
    metadata["nt"] = original_nt - args.time_start
    metadata["t_min"] = t_min + args.time_start * dt
    metadata["time_slice"] = {
        "source_start": args.time_start,
        "source_nt": original_nt,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
