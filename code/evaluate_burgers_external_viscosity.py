"""Zero-shot evaluate the Burgers operator on an official unseen viscosity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from baselines_1d_burgers import FNO1d
from burgers_common import relative_l2
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from pde_residual_1d import normalized_pde_residual


SOURCE = (
    ROOT
    / "datasets"
    / "external"
    / "burgers_nu0.1"
    / "test.npz"
)
OUTPUT = (
    ROOT
    / "analysis_plots_clean"
    / "burgers_external_nu0.1.json"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fields() -> torch.Tensor:
    with np.load(SOURCE, allow_pickle=False) as data:
        return torch.from_numpy(data["fields"]).float().to(DEVICE)


def predict_pg(path: Path, fields: torch.Tensor) -> torch.Tensor:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    valid = set(OperatorConfig.__dataclass_fields__)
    operator = {
        key: value
        for key, value in checkpoint["operator"].items()
        if key in valid
    }
    model = CoordAttentionOperator(OperatorConfig(**operator)).to(DEVICE).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    predictions = []
    with torch.no_grad():
        for start in range(0, fields.size(0), 64):
            predictions.append(
                model.forward_grid(fields[start : start + 64, :, 0, 0])
            )
    return torch.cat(predictions, dim=0)


def predict_fno(path: Path, fields: torch.Tensor) -> torch.Tensor:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = FNO1d(
        config["modes"],
        config["width"],
        config["layers"],
        fields.size(2),
    ).to(DEVICE).eval()
    model.load_state_dict(checkpoint["state_dict"])
    predictions = []
    with torch.no_grad():
        for start in range(0, fields.size(0), 64):
            predictions.append(model(fields[start : start + 64, :, 0, 0]))
    return torch.cat(predictions, dim=0)


def main() -> None:
    fields = load_fields()
    result = {}
    checkpoints = {
        "pg_derivative_refiner": (
            ROOT
            / "experiments"
            / "pg_physics_hpo"
            / "refiner_h18_k11_w03_deriv2k5_4k_budget02_s7"
            / "model.pt",
            predict_pg,
        ),
        "fno": (
            ROOT
            / "experiments"
            / "pino_formal"
            / "burgers"
            / "w0001_s7_24k"
            / "model.pt",
            predict_fno,
        ),
    }
    for name, (path, predictor) in checkpoints.items():
        prediction = predictor(path, fields)
        result[name] = {
            "test_l2": float(relative_l2(prediction, fields[..., 0])),
            "pde_relative_rms": float(
                normalized_pde_residual(
                    prediction,
                    "burgers",
                    1.0,
                    nu=0.1,
                )[1]
            ),
        }
    result["truth_pde_relative_rms"] = float(
        normalized_pde_residual(
            fields[..., 0],
            "burgers",
            1.0,
            nu=0.1,
        )[1]
    )
    result["protocol"] = (
        "Official PDEBench Burgers Nu=0.1 test split, zero-shot from models "
        "trained only on Nu=0.01. No test tuning."
    )
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
