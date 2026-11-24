"""Compare time-averaged radial spectra for trained 2D operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_2d_models import load_model, predict
from operator_2d_common import load_npz_split, load_spec


def radial_spectrum(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fields [N, C, X, Y, T] -> mean radial power spectrum."""
    samples, channels, height, width, times = fields.shape
    window_x = np.hanning(height).astype(np.float32)
    window_y = np.hanning(width).astype(np.float32)
    window = window_x[:, None] * window_y[None, :]
    spectrum = np.fft.fft2(
        fields * window[None, None, :, :, None],
        axes=(2, 3),
    )
    power = np.abs(spectrum) ** 2
    power = power.mean(axis=(0, 1, 4))
    frequency_x = np.fft.fftfreq(height) * height
    frequency_y = np.fft.fftfreq(width) * width
    grid_x, grid_y = np.meshgrid(
        frequency_x,
        frequency_y,
        indexing="ij",
    )
    radius = np.sqrt(grid_x**2 + grid_y**2).astype(int)
    max_radius = int(np.ceil(radius.max()))
    radial = np.zeros(max_radius + 1, dtype=np.float64)
    counts = np.zeros(max_radius + 1, dtype=np.int64)
    for index in range(max_radius + 1):
        mask = radius == index
        radial[index] = power[mask].mean()
        counts[index] = mask.sum()
    return radial, counts


def high_frequency_ratio(
    radial: np.ndarray,
    cutoff: float = 0.25,
) -> float:
    total = radial.sum()
    if total <= 0.0:
        return 0.0
    start = int(round((len(radial) - 1) * cutoff))
    return float(radial[start:].sum() / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--pg-checkpoint", type=Path, required=True)
    parser.add_argument("--fno-checkpoint", type=Path, required=True)
    parser.add_argument("--deeponet-checkpoint", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_spec(args.data_dir / "metadata.json")
    inputs, targets = load_npz_split(args.data_dir / "test.npz")
    model_specs = [
        ("PG-MS-LR-LPO-2D", args.pg_checkpoint, "#087F8C"),
        ("FNO-2D", args.fno_checkpoint, "#D97706"),
    ]
    if args.deeponet_checkpoint is not None:
        model_specs.append(
            ("DeepONet-2D", args.deeponet_checkpoint, "#B91C1C")
        )

    fields = {"Target": targets.numpy()}
    for label, checkpoint, _ in model_specs:
        model, _ = load_model(checkpoint, spec, device)
        fields[label] = predict(model, inputs, device).numpy()

    spectra = {}
    summary = {}
    for label, values in fields.items():
        radial, counts = radial_spectrum(values)
        spectra[label] = radial
        summary[label] = {
            "high_frequency_ratio": high_frequency_ratio(radial),
            "total_power": float(radial.sum()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "radial_spectrum.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_label,
                "radial_power": {
                    label: values.tolist()
                    for label, values in spectra.items()
                },
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for label, _, color in model_specs:
        axis.plot(
            np.arange(len(spectra[label])),
            np.maximum(spectra[label], 1.0e-16),
            label=label,
            color=color,
            linewidth=2,
        )
    axis.plot(
        np.arange(len(spectra["Target"])),
        np.maximum(spectra["Target"], 1.0e-16),
        label="Target",
        color="#17212B",
        linewidth=2.5,
        linestyle="--",
    )
    axis.set_yscale("log")
    axis.set_xlabel("radial wavenumber")
    axis.set_ylabel("time-averaged power")
    axis.set_title(f"{args.dataset_label}: radial spectrum")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "radial_spectrum.png",
        dpi=220,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
