"""Convert the official PDEBench 1D Burgers HDF5 into project parquet files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PARAMETERS = "Burgers_Sols_Nu0.01"


def nested_tensor(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    flattened = pa.array(values.reshape(-1), type=pa.float32())
    channels = pa.FixedSizeListArray.from_arrays(flattened, 1)
    time = pa.FixedSizeListArray.from_arrays(channels, values.shape[2])
    return pa.FixedSizeListArray.from_arrays(time, values.shape[1])


def write_split(
    source: h5py.Dataset,
    output: Path,
    start: int,
    count: int,
    batch_size: int,
) -> None:
    schema = pa.schema([
        pa.field("parameters", pa.string()),
        pa.field(
            "tensor",
            pa.list_(
                pa.list_(
                    pa.list_(pa.float32(), 1),
                    101,
                ),
                256,
            ),
        ),
    ])
    stop = start + count
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            raw = np.asarray(source[offset:end], dtype=np.float32)
            fields = raw[:, ::2, ::4].transpose(0, 2, 1)[..., None]
            table = pa.Table.from_arrays(
                [
                    pa.array([PARAMETERS] * fields.shape[0], type=pa.string()),
                    nested_tensor(fields),
                ],
                schema=schema,
            )
            writer.write_table(table)
            print(f"{output.name}: {end - start}/{count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=1000)
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--dev-start", type=int, default=9000)
    parser.add_argument("--dev-count", type=int, default=1000)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as source:
        tensor = source["tensor"]
        splits = {
            "train": (args.train_start, args.train_count),
            "dev": (args.dev_start, args.dev_count),
            "test": (args.test_start, args.test_count),
        }
        for name, (start, count) in splits.items():
            if start < 0 or count <= 0 or start + count > tensor.shape[0]:
                raise ValueError(f"invalid {name} range: {start}:{start + count}")
            output = args.output_dir / f"burgers_nu0.01_{name}.parquet"
            if name == "train":
                output = args.output_dir / "burgers_nu0.01_train_000.parquet"
            write_split(tensor, output, start, count, args.batch_size)

    metadata = {
        "source": str(args.input),
        "source_shape": [10000, 201, 1024],
        "source_nu": 0.01,
        "downsample": {"time_stride": 2, "x_stride": 4},
        "splits": {
            "train": [args.train_start, args.train_start + args.train_count],
            "dev": [args.dev_start, args.dev_start + args.dev_count],
            "test": [args.test_start, args.test_start + args.test_count],
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
