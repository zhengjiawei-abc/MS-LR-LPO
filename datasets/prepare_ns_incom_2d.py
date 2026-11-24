"""Build forced incompressible NS windows from PDEBench 2D NS files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def build_windows(
    velocity: np.ndarray,
    force: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    targets = []
    for start in range(0, velocity.shape[0] - window + 1, stride):
        trajectory = velocity[start : start + window]
        initial_velocity = trajectory[0]
        model_input = np.concatenate([initial_velocity, force], axis=-1)
        inputs.append(model_input.transpose(2, 0, 1))
        targets.append(trajectory.transpose(3, 1, 2, 0))
    return (
        np.stack(inputs).astype(np.float16),
        np.stack(targets).astype(np.float16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-stride", type=int, default=8)
    parser.add_argument("--y-stride", type=int, default=8)
    parser.add_argument("--window", type=int, default=41)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--eval-stride", type=int, default=20)
    parser.add_argument("--train-trajectories", default="0,1")
    parser.add_argument("--dev-trajectories", default="2")
    parser.add_argument("--test-trajectories", default="3")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    train_indices = [
        int(value)
        for value in args.train_trajectories.split(",")
        if value.strip()
    ]
    dev_indices = [
        int(value)
        for value in args.dev_trajectories.split(",")
        if value.strip()
    ]
    test_indices = [
        int(value)
        for value in args.test_trajectories.split(",")
        if value.strip()
    ]

    with h5py.File(args.input, "r") as source:
        velocity_dataset = source["velocity"]
        force_dataset = source["force"]
        time_dataset = source["t"]
        shape = tuple(velocity_dataset.shape)
        force_shape = tuple(force_dataset.shape)
        if len(shape) != 5 or shape[-1] != 2:
            raise ValueError(f"unsupported velocity shape {shape}")
        if (
            force_shape[0] != shape[0]
            or force_shape[1:3] != shape[2:4]
            or force_shape[-1] != 2
        ):
            raise ValueError(
                f"force shape {force_shape} does not match velocity {shape}"
            )
        time = np.asarray(time_dataset[...], dtype=np.float32)
        dt = float(time[0, 1] - time[0, 0])
        inspection = {
            "shape": list(shape),
            "force_shape": list(force_shape),
            "trajectories": shape[0],
            "time_steps": shape[1],
            "grid": [shape[2], shape[3]],
            "channels": shape[4],
            "dt": dt,
            "time_range": [float(time.min()), float(time.max())],
            "attributes": {
                str(key): str(value)
                for key, value in source.attrs.items()
            },
        }
        print(json.dumps(inspection, indent=2), flush=True)
        if args.inspect_only:
            return

        for name, indices in (
            ("train", train_indices),
            ("dev", dev_indices),
            ("test", test_indices),
        ):
            if not indices or min(indices) < 0 or max(indices) >= shape[0]:
                raise ValueError(
                    f"invalid {name} trajectories: {indices}"
                )
        if set(train_indices) & (set(dev_indices) | set(test_indices)):
            raise ValueError("train trajectories overlap dev/test")
        if set(dev_indices) & set(test_indices):
            raise ValueError("dev/test trajectories overlap")
        if args.window < 2 or args.window > shape[1]:
            raise ValueError("invalid window length")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        nx = len(range(0, shape[2], args.x_stride))
        ny = len(range(0, shape[3], args.y_stride))
        split_counts = {}
        for split, indices, stride in (
            ("train", train_indices, args.train_stride),
            ("dev", dev_indices, args.eval_stride),
            ("test", test_indices, args.eval_stride),
        ):
            inputs = []
            targets = []
            for trajectory_index in indices:
                velocity = np.asarray(
                    velocity_dataset[
                        trajectory_index,
                        :,
                        :: args.x_stride,
                        :: args.y_stride,
                        :,
                    ],
                    dtype=np.float16,
                )
                force = np.asarray(
                    force_dataset[
                        trajectory_index,
                        :: args.x_stride,
                        :: args.y_stride,
                        :,
                    ],
                    dtype=np.float16,
                )
                split_inputs, split_targets = build_windows(
                    velocity,
                    force,
                    args.window,
                    stride,
                )
                inputs.append(split_inputs)
                targets.append(split_targets)
                print(
                    f"{split}: trajectory {trajectory_index} "
                    f"{split_inputs.shape[0]} windows",
                    flush=True,
                )
            split_inputs = np.concatenate(inputs, axis=0)
            split_targets = np.concatenate(targets, axis=0)
            np.savez(
                args.output_dir / f"{split}.npz",
                inputs=split_inputs,
                targets=split_targets,
            )
            split_counts[split] = {
                "windows": int(split_inputs.shape[0]),
                "trajectories": indices,
            }

    metadata = {
        "source": str(args.input),
        "dataset": "PDEBench 2D incompressible NS",
        "physics": "incompressible_ns",
        "source_shape": list(shape),
        "input_fields": ["u0", "v0", "force_x", "force_y"],
        "target_fields": ["u", "v"],
        "input_channels": 4,
        "channels": [0, 1],
        "nx": nx,
        "ny": ny,
        "nt": args.window,
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "t_min": 0.0,
        "t_max": dt * (args.window - 1),
        "downsample": {
            "x_stride": args.x_stride,
            "y_stride": args.y_stride,
        },
        "window": {
            "length": args.window,
            "train_stride": args.train_stride,
            "eval_stride": args.eval_stride,
        },
        "splits": split_counts,
        "source_attributes": inspection["attributes"],
        "storage": "float16 NPZ; force-conditioned state-to-future windows",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
