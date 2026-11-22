"""Evaluate individual multi-scale skip branches from the corrected model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from burgers_common import load_split, relative_l2
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from train_coord_attention import predict_full


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
RUN = (
    ROOT
    / "experiments"
    / "outputs_burgers_dataset"
    / "official_n8000"
    / "latent_pde_best_skip_corrected"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, fields):
    return relative_l2(predict_full(model, fields, DEVICE), fields[..., 0])


def main() -> None:
    checkpoint = torch.load(RUN / "model.pt", map_location=DEVICE, weights_only=False)
    model = CoordAttentionOperator(
        OperatorConfig(**checkpoint["operator"])
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dev = load_split(DATA / "burgers_nu0.01_dev.parquet", 1000).to(DEVICE)
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)

    baseline = {"dev": evaluate(model, dev), "test": evaluate(model, test)}
    state = copy.deepcopy(model.state_dict())
    scales = []
    for index, projection in enumerate(model.grid_decoder.skip_projs):
        model.load_state_dict(state)
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
        scales.append(
            {
                "removed_scale": index,
                "dev": evaluate(model, dev),
                "test": evaluate(model, test),
            }
        )

    model.load_state_dict(state)
    for projection in model.grid_decoder.skip_projs:
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
    without_all = {"dev": evaluate(model, dev), "test": evaluate(model, test)}
    model.load_state_dict(state)

    gates = torch.sigmoid(
        model.grid_decoder.skip_gate(
            torch.zeros(1, 64, device=DEVICE)
        )
    )
    result = {
        "baseline": baseline,
        "remove_scale": scales,
        "without_all_skips": without_all,
    }
    print(json.dumps(result, indent=2), flush=True)

    chosen = min(scales, key=lambda row: row["dev"])
    if chosen["dev"] < baseline["dev"]:
        projection = model.grid_decoder.skip_projs[chosen["removed_scale"]]
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
        torch.save(
            {"state_dict": model.state_dict(), "operator": checkpoint["operator"]},
            RUN / "model_corrected.pt",
        )
        result["corrected_scale"] = chosen["removed_scale"]
    (RUN / "scale_diagnostics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
