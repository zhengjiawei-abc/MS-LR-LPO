"""Find the source index of a parquet sample in a PDEBench HDF5 file."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def digest(array: np.ndarray) -> str:
    return hashlib.sha1(
        np.ascontiguousarray(array, dtype=np.float32).tobytes()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["tensor"])
    target = np.asarray(table.column("tensor")[args.row].as_py(), dtype=np.float32)
    target_hash = digest(target)
    print(f"target {target.shape} {target_hash}", flush=True)
    target_norm = target - target.mean()
    target_norm /= max(float(target_norm.std()), 1e-8)
    best = []

    with h5py.File(args.hdf5, "r") as handle:
        data = handle["tensor"]
        for start in range(0, data.shape[0], args.batch):
            stop = min(start + args.batch, data.shape[0])
            fields = np.asarray(data[start:stop], dtype=np.float32)
            samples = fields[:, ::2, ::4].transpose(0, 2, 1)[..., None]
            for offset, sample in enumerate(samples):
                if digest(sample) == target_hash:
                    print(f"match {start + offset}", flush=True)
                    return
            normalized = samples - samples.mean(axis=(1, 2, 3), keepdims=True)
            normalized /= np.maximum(
                normalized.std(axis=(1, 2, 3), keepdims=True),
                1e-8,
            )
            errors = np.square(normalized[..., 0] - target_norm[..., 0]).mean(
                axis=(1, 2)
            )
            for local in np.argsort(errors)[:3]:
                best.append((float(errors[local]), start + int(local)))
            best = sorted(best)[:5]
            print(f"checked {stop}/{data.shape[0]}", flush=True)
    print("nearest normalized:", best, flush=True)


if __name__ == "__main__":
    main()
