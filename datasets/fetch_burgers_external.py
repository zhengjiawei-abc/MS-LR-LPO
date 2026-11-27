"""Fetch a PDEBench Burgers test split through HTTP ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from http_hdf5 import open_http_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--t-stride", type=int, default=2)
    parser.add_argument("--x-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_http_hdf5(args.url) as source:
        tensor = source["tensor"]
        source_shape = list(tensor.shape)
        time = np.asarray(source["t-coordinate"][...], dtype=np.float32)
        x = np.asarray(source["x-coordinate"][...], dtype=np.float32)
        fields = []
        stop_all = args.start + args.samples
        for start in range(args.start, stop_all, args.batch_size):
            stop = min(start + args.batch_size, stop_all)
            raw = np.asarray(tensor[start:stop], dtype=np.float32)
            selected = raw[:, :: args.t_stride, :: args.x_stride]
            fields.append(selected.transpose(0, 2, 1)[..., None])
            print(f"fetched {stop - args.start}/{args.samples}", flush=True)
        data = np.concatenate(fields, axis=0)
        attributes = {
            key: float(value)
            for key, value in source.attrs.items()
            if np.isscalar(value)
        }

    np.savez(args.output, fields=data)
    metadata = {
        "source": args.url,
        "source_shape": source_shape,
        "samples": int(data.shape[0]),
        "source_start": args.start,
        "output_shape": list(data.shape[1:]),
        "time_range": [float(time[0]), float(time[-1])],
        "x_range": [float(x[0]), float(x[-1])],
        "downsample": {
            "t_stride": args.t_stride,
            "x_stride": args.x_stride,
        },
        "attributes": attributes,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
