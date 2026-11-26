"""Zero-shot evaluate the beta=1 learned refiner across official betas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from burgers_common import load_split, relative_l2
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from pde_residual_1d import normalized_pde_residual
from run_advection_beta_queue import fixed_transport_prediction


BETAS = {
    0.1: ("advection_beta01", "pdebench_advection_beta01_official"),
    0.2: ("advection_beta02", "pdebench_advection_beta02_official"),
    0.4: ("advection_beta04", "pdebench_advection_beta04_official"),
    0.7: ("advection_beta07", "pdebench_advection_beta07_official"),
    1.0: ("advection_beta1", "pdebench_advection_beta1_official"),
    2.0: ("advection_beta20", "pdebench_advection_beta20_official"),
    4.0: ("advection_beta40", "pdebench_advection_beta40_official"),
    7.0: ("advection_beta70", "pdebench_advection_beta70_official"),
}
SOURCE = (
    ROOT
    / "experiments"
    / "pg_advection_output_refiner"
    / "w0008_budget30_s7_4k"
    / "model.pt"
)
OUTPUT = (
    ROOT
    / "analysis_plots_clean"
    / "advection_refiner_generalization.json"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    checkpoint = torch.load(SOURCE, map_location=DEVICE, weights_only=False)
    valid = set(OperatorConfig.__dataclass_fields__)
    base_operator = {
        key: value
        for key, value in checkpoint["operator"].items()
        if key in valid
    }
    summary = {}
    for beta, (prefix, data_name) in BETAS.items():
        operator = dict(base_operator)
        operator["grid_transport_velocity"] = beta
        operator["grid_transport_blend"] = 0.0
        operator["grid_physics_equation"] = "advection"
        operator["grid_physics_beta"] = beta
        model = CoordAttentionOperator(
            OperatorConfig(**operator)
        ).to(DEVICE).eval()
        state = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if not key.startswith("transport_")
        }
        model.load_state_dict(state, strict=False)
        test = load_split(
            ROOT
            / "datasets"
            / data_name
            / f"{prefix}_test.parquet",
            1000,
        ).to(DEVICE)
        predictions = []
        with torch.no_grad():
            for start in range(0, test.size(0), 64):
                predictions.append(
                    model.forward_grid(test[start : start + 64, :, 0, 0])
                )
        prediction = torch.cat(predictions, dim=0)
        fixed = fixed_transport_prediction(test[:, :, 0, 0], beta)
        summary[f"beta{beta:g}"] = {
            "fixed_l2": float(relative_l2(fixed, test[..., 0])),
            "refiner_l2": float(relative_l2(prediction, test[..., 0])),
            "fixed_pde_relative_rms": float(
                normalized_pde_residual(fixed, "advection", beta)[1]
            ),
            "refiner_pde_relative_rms": float(
                normalized_pde_residual(
                    prediction,
                    "advection",
                    beta,
                )[1]
            ),
        }
        print(summary[f"beta{beta:g}"], flush=True)

    OUTPUT.write_text(
        json.dumps(
            {
                "protocol": (
                    "Zero-shot transfer of the beta=1, seed-7 output refiner. "
                    "Only the transport velocity and PDE beta are changed; "
                    "no test-set tuning or retraining is performed."
                ),
                "results": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
