"""Evaluate and visualize 2D operator checkpoints on a held-out NPZ split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines_2d import FNO2D, FNO2DConfig
from coord_attention_operator_2d import (
    CoordAttentionOperator2D,
    Operator2DConfig,
)
from operator_2d_common import (
    divergence_relative_loss,
    load_npz_split,
    load_spec,
    relative_l2,
    space_time_coordinates,
)


def load_model(path: Path, spec, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    name = checkpoint["model_name"]
    config = checkpoint["config"]
    if name == "pg2d":
        model = CoordAttentionOperator2D(
            Operator2DConfig(**config),
            spec.nx,
            spec.ny,
            spec.nt,
        )
    elif name == "fno2d":
        model = FNO2D(
            FNO2DConfig(**config),
            spec.nx,
            spec.ny,
            spec.nt,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), name


@torch.no_grad()
def predict(model, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    prediction = []
    for start in range(0, inputs.size(0), 2):
        stop = min(start + 2, inputs.size(0))
        prediction.append(
            model.forward_grid(inputs[start:stop].to(device)).cpu()
        )
    return torch.cat(prediction)


def per_time_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> list[float]:
    values = []
    for time_index in range(target.size(-1)):
        difference = prediction[..., time_index] - target[..., time_index]
        error = torch.linalg.vector_norm(difference.flatten(1), dim=1)
        scale = torch.linalg.vector_norm(
            target[..., time_index].flatten(1),
            dim=1,
        ).clamp_min(1.0e-8)
        values.append(float((error / scale).mean()))
    return values


def plot_time_error(
    pg_values: list[float],
    fno_values: list[float],
    time_values: np.ndarray,
    dataset_label: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.plot(
        time_values,
        pg_values,
        marker="o",
        linewidth=2,
        label="PG-MS-LR-LPO-2D",
    )
    axis.plot(
        time_values,
        fno_values,
        marker="s",
        linewidth=2,
        label="FNO-2D",
    )
    axis.set_yscale("log")
    axis.set_xlabel("physical time")
    axis.set_ylabel("relative L2")
    axis.set_title(f"{dataset_label}: error over time")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_sample(
    target: np.ndarray,
    predictions: list[tuple[str, np.ndarray]],
    sample_index: int,
    channel: int,
    time_values: np.ndarray,
    dataset_label: str,
    output: Path,
) -> None:
    time_points = target.shape[-1]
    columns = sorted(
        set(
            int(round(index))
            for index in np.linspace(
                0,
                time_points - 1,
                min(5, time_points),
            )
        )
    )
    rows = 1 + 2 * len(predictions)
    figure, axes = plt.subplots(
        rows,
        len(columns),
        figsize=(3.0 * len(columns), 2.65 * rows),
        constrained_layout=True,
    )
    if rows == 1:
        axes = axes[None, :]
    scale_min = min(
        float(target[sample_index, channel].min()),
        *(float(value[sample_index, channel].min()) for _, value in predictions),
    )
    scale_max = max(
        float(target[sample_index, channel].max()),
        *(float(value[sample_index, channel].max()) for _, value in predictions),
    )
    images = []
    for column, time_index in enumerate(columns):
        image = axes[0, column].imshow(
            target[sample_index, channel, :, :, time_index],
            cmap="viridis",
            vmin=scale_min,
            vmax=scale_max,
            origin="lower",
        )
        images.append(image)
        time_label = f"{time_values[time_index]:.2f}"
        axes[0, column].set_title(f"target t={time_label}")
        axes[0, column].axis("off")
        for prediction_index, (label, prediction) in enumerate(predictions):
            field = prediction[sample_index, channel, :, :, time_index]
            image = axes[1 + 2 * prediction_index, column].imshow(
                field,
                cmap="viridis",
                vmin=scale_min,
                vmax=scale_max,
                origin="lower",
            )
            images.append(image)
            axes[1 + 2 * prediction_index, column].set_title(
                f"{label} t={time_label}"
            )
            axes[1 + 2 * prediction_index, column].axis("off")
            error = np.abs(
                field
                - target[sample_index, channel, :, :, time_index]
            )
            error_image = axes[
                2 + 2 * prediction_index,
                column,
            ].imshow(error, cmap="magma", origin="lower")
            images.append(error_image)
            axes[2 + 2 * prediction_index, column].set_title(
                f"|{label} - target|"
            )
            axes[2 + 2 * prediction_index, column].axis("off")
    figure.colorbar(
        images[0],
        ax=axes[0, :],
        shrink=0.8,
        label="field",
    )
    figure.colorbar(
        images[-1],
        ax=axes[-1, :],
        shrink=0.8,
        label="absolute error",
    )
    figure.suptitle(
        f"{dataset_label} | sample {sample_index} | channel {channel}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pg-checkpoint", type=Path, required=True)
    parser.add_argument("--fno-checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_label = metadata.get("dataset", "PDEBench 2D PDE")
    spec = load_spec(metadata_path)
    incompressible = (
        metadata.get("physics") == "incompressible_ns"
        and spec.channels >= 2
    )
    inputs, targets = load_npz_split(args.data_dir / "test.npz")
    time_values = np.linspace(spec.t_min, spec.t_max, spec.nt)
    pg, _ = load_model(args.pg_checkpoint, spec, device)
    fno, _ = load_model(args.fno_checkpoint, spec, device)
    pg_prediction = predict(pg, inputs, device)
    fno_prediction = predict(fno, inputs, device)
    coordinates = space_time_coordinates(spec, torch.device("cpu"))

    metrics = {
        "pg2d": {
            "relative_l2": relative_l2(pg_prediction, targets),
            "divergence_relative": float(
                divergence_relative_loss(pg_prediction, coordinates)
            )
            if incompressible
            else None,
            "parameters": sum(value.numel() for value in pg.parameters()),
            "time_relative_l2": per_time_relative_l2(
                pg_prediction,
                targets,
            ),
        },
        "fno2d": {
            "relative_l2": relative_l2(fno_prediction, targets),
            "divergence_relative": float(
                divergence_relative_loss(fno_prediction, coordinates)
            )
            if incompressible
            else None,
            "parameters": sum(value.numel() for value in fno.parameters()),
            "time_relative_l2": per_time_relative_l2(
                fno_prediction,
                targets,
            ),
        },
        "test_samples": inputs.size(0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    plot_time_error(
        metrics["pg2d"]["time_relative_l2"],
        metrics["fno2d"]["time_relative_l2"],
        time_values,
        dataset_label,
        args.output_dir / "time_error_curve.png",
    )

    predictions = [
        ("PG-MS-LR-LPO-2D", pg_prediction.numpy()),
        ("FNO-2D", fno_prediction.numpy()),
    ]
    target = targets.numpy()
    for sample_index in range(min(args.samples, inputs.size(0))):
        for channel in range(spec.channels):
            plot_sample(
                target,
                predictions,
                sample_index,
                channel,
                time_values,
                dataset_label,
                args.output_dir
                / f"sample_{sample_index:02d}_channel_{channel}.png",
            )


if __name__ == "__main__":
    main()
