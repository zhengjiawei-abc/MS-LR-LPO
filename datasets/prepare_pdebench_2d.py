"""Convert selected PDEBench 2D trajectories into compact project NPZ splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def axis_length(shape: tuple[int, ...], axis: int) -> int:
    return int(shape[axis])


def detect_layout(
    shape: tuple[int, ...],
    time_count: int,
    x_count: int,
    y_count: int,
) -> str:
    if len(shape) == 4:
        shape = shape + (1,)
    if len(shape) != 5:
        raise ValueError(f"expected a 4D or 5D tensor, got {shape}")
    if shape[0] < 2:
        raise ValueError(f"first axis is not a sample axis: {shape}")
    candidates = []
    for axis in range(1, 5):
        if axis_length(shape, axis) == time_count:
            candidates.append(axis)
    if len(candidates) != 1:
        raise ValueError(
            f"cannot identify time axis from shape {shape}, t={time_count}"
        )
    time_axis = candidates[0]
    spatial_axes = [axis for axis in range(1, 5) if axis != time_axis]
    lengths = sorted(shape[axis] for axis in spatial_axes[:2])
    if lengths != sorted([x_count, y_count]):
        raise ValueError(
            f"cannot match spatial axes from shape {shape}, "
            f"x={x_count}, y={y_count}"
        )
    channel_axis = spatial_axes[-1]
    if shape[channel_axis] > 8:
        raise ValueError(
            f"cannot identify channel axis from shape {shape}"
        )
    if time_axis == 1 and channel_axis == 4:
        return "n,t,x,y,c"
    if time_axis == 3 and channel_axis == 4:
        return "n,x,y,t,c"
    raise ValueError(
        f"unsupported axis order {shape}; expected n,t,x,y,c or n,x,y,t,c"
    )


def select_tensor(
    tensor: h5py.Dataset,
    layout: str,
    start: int,
    stop: int,
    x_stride: int,
    y_stride: int,
    t_stride: int,
    channel_indices: list[int],
) -> np.ndarray:
    if layout == "n,t,x,y,c":
        raw = np.asarray(
            tensor[
                start:stop,
                ::t_stride,
                ::x_stride,
                ::y_stride,
                channel_indices,
            ],
            dtype=np.float32,
        )
    elif layout == "n,x,y,t,c":
        raw = np.asarray(
            tensor[
                start:stop,
                ::x_stride,
                ::y_stride,
                ::t_stride,
                channel_indices,
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unsupported layout: {layout}")
    return np.ascontiguousarray(raw.transpose(0, 4, 2, 3, 1))


def write_split(
    tensor: h5py.Dataset,
    layout: str,
    output: Path,
    start: int,
    count: int,
    x_stride: int,
    y_stride: int,
    t_stride: int,
    channel_indices: list[int],
    batch_size: int,
) -> tuple[int, int, int]:
    fields = None
    shape = None
    for offset in range(start, start + count, batch_size):
        stop = min(offset + batch_size, start + count)
        batch = select_tensor(
            tensor,
            layout,
            offset,
            stop,
            x_stride,
            y_stride,
            t_stride,
            channel_indices,
        )
        if fields is None:
            shape = batch.shape
            fields = np.empty(
                (count, *batch.shape[1:]),
                dtype=np.float16,
            )
        local_start = offset - start
        fields[local_start : local_start + batch.shape[0]] = batch.astype(
            np.float16
        )
        print(
            f"{output.name}: {stop - start}/{count}",
            flush=True,
        )
    if fields is None or shape is None:
        raise ValueError("empty split")
    np.savez(
        output,
        inputs=fields[..., 0],
        targets=fields,
    )
    return shape[2], shape[3], shape[4]


def sample_keys(source: h5py.File) -> list[str]:
    keys = [key for key in source.keys() if key.isdigit()]
    return sorted(keys, key=int)


def read_group_sample(
    source: h5py.File,
    key: str,
    x_stride: int,
    y_stride: int,
    t_stride: int,
    channel_indices: list[int],
) -> np.ndarray:
    raw = np.asarray(
        source[key]["data"][
            ::t_stride,
            ::x_stride,
            ::y_stride,
            channel_indices,
        ],
        dtype=np.float32,
    )
    if raw.ndim != 4:
        raise ValueError(f"{key}/data has unsupported shape {raw.shape}")
    return np.ascontiguousarray(raw.transpose(3, 1, 2, 0))


def write_group_split(
    source: h5py.File,
    keys: list[str],
    output: Path,
    x_stride: int,
    y_stride: int,
    t_stride: int,
    channel_indices: list[int],
) -> tuple[int, int, int]:
    first = read_group_sample(
        source,
        keys[0],
        x_stride,
        y_stride,
        t_stride,
        channel_indices,
    )
    fields = np.empty((len(keys), *first.shape), dtype=np.float16)
    fields[0] = first.astype(np.float16)
    for index, key in enumerate(keys[1:], start=1):
        fields[index] = read_group_sample(
            source,
            key,
            x_stride,
            y_stride,
            t_stride,
            channel_indices,
        ).astype(np.float16)
        if (index + 1) % 16 == 0 or index + 1 == len(keys):
            print(
                f"{output.name}: {index + 1}/{len(keys)}",
                flush=True,
            )
    np.savez(
        output,
        inputs=fields[..., 0],
        targets=fields,
    )
    return fields.shape[2], fields.shape[3], fields.shape[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="diff_react_2d")
    parser.add_argument("--x-stride", type=int, default=2)
    parser.add_argument("--y-stride", type=int, default=2)
    parser.add_argument("--t-stride", type=int, default=2)
    parser.add_argument("--channels", default="0,1")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--dev-start", type=int, default=800)
    parser.add_argument("--dev-count", type=int, default=100)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    channel_indices = [
        int(value) for value in args.channels.split(",") if value.strip()
    ]
    if not channel_indices:
        raise ValueError("at least one channel is required")

    with h5py.File(args.input, "r") as source:
        grouped = "tensor" not in source
        tensor = None if grouped else source["tensor"]
        if grouped:
            keys = sample_keys(source)
            if not keys:
                raise RuntimeError("no numbered PDEBench trajectories found")
            first_group = source[keys[0]]
            data_shape = tuple(first_group["data"].shape)
            if len(data_shape) != 4:
                raise ValueError(
                    f"unsupported grouped data shape {data_shape}"
                )
            time = np.asarray(
                first_group["grid"]["t"][...],
                dtype=np.float32,
            )
            x = np.asarray(
                first_group["grid"]["x"][...],
                dtype=np.float32,
            )
            y = np.asarray(
                first_group["grid"]["y"][...],
                dtype=np.float32,
            )
            if data_shape != (len(time), len(x), len(y), data_shape[-1]):
                raise ValueError(
                    f"grouped data shape {data_shape} does not match "
                    f"t={len(time)}, x={len(x)}, y={len(y)}"
                )
            layout = "group,t,x,y,c"
            total = len(keys)
            attributes = {
                str(key): str(value)
                for key, value in first_group.attrs.items()
            }
        else:
            time = np.asarray(
                source["t-coordinate"][...],
                dtype=np.float32,
            )
            x = np.asarray(
                source["x-coordinate"][...],
                dtype=np.float32,
            )
            y = np.asarray(
                source["y-coordinate"][...],
                dtype=np.float32,
            )
            layout = detect_layout(
                tuple(tensor.shape),
                len(time),
                len(x),
                len(y),
            )
            total = int(tensor.shape[0])
            data_shape = tuple(tensor.shape)
            attributes = {
                str(key): str(value)
                for key, value in tensor.attrs.items()
            }
        inspection = {
            "shape": list(data_shape),
            "layout": layout,
            "samples": total,
            "time_points": len(time),
            "x_points": len(x),
            "y_points": len(y),
            "x_range": [float(x[0]), float(x[-1])],
            "y_range": [float(y[0]), float(y[-1])],
            "t_range": [float(time[0]), float(time[-1])],
            "attributes": attributes,
        }
        print(json.dumps(inspection, indent=2), flush=True)
        if args.inspect_only:
            return

        splits = {
            "train": (args.train_start, args.train_count),
            "dev": (args.dev_start, args.dev_count),
            "test": (args.test_start, args.test_count),
        }
        for split, (start, count) in splits.items():
            if start < 0 or count <= 0 or start + count > total:
                raise ValueError(
                    f"invalid {split} range {start}:{start + count} "
                    f"for {total} samples"
                )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_shape = None
        for split, (start, count) in splits.items():
            output = args.output_dir / f"{split}.npz"
            if grouped:
                output_shape = write_group_split(
                    source,
                    keys[start : start + count],
                    output,
                    args.x_stride,
                    args.y_stride,
                    args.t_stride,
                    channel_indices,
                )
            else:
                output_shape = write_split(
                    tensor,
                    layout,
                    output,
                    start,
                    count,
                    args.x_stride,
                    args.y_stride,
                    args.t_stride,
                    channel_indices,
                    args.batch_size,
                )

    if output_shape is None:
        raise RuntimeError("no split was written")
    nx, ny, nt = output_shape
    metadata = {
        "source": str(args.input),
        "name": args.name,
        "dataset": "PDEBench 2D diffusion-reaction",
        "source_shape": inspection["shape"],
        "source_layout": layout,
        "source_ranges": {
            "x": inspection["x_range"],
            "y": inspection["y_range"],
            "t": inspection["t_range"],
        },
        "downsample": {
            "x_stride": args.x_stride,
            "y_stride": args.y_stride,
            "t_stride": args.t_stride,
        },
        "channels": channel_indices,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "x_min": inspection["x_range"][0],
        "x_max": inspection["x_range"][1],
        "y_min": inspection["y_range"][0],
        "y_max": inspection["y_range"][1],
        "t_min": inspection["t_range"][0],
        "t_max": inspection["t_range"][1],
        "splits": {
            name: [start, start + count]
            for name, (start, count) in splits.items()
        },
        "storage": "float16 NPZ; input is first time slice",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
