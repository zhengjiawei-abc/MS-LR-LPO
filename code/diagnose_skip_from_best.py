"""Load the best context-grid checkpoint and diagnose skip and decoder layers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from burgers_common import load_split, relative_l2, set_seed
from coord_attention_operator import CoordAttentionOperator, OperatorConfig
from train_coord_attention import predict_full


DATA = ROOT / "datasets" / "pdebench_burgers_official"
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
CHECKPOINT = RUNS / "latent_pde_context_grid_s7_24k" / "model.pt"
OUTPUT = RUNS / "latent_pde_best_skip_corrected"
SEED = 7
STEPS = 2000
BATCH = 64


def evaluate(model, fields, device):
    return relative_l2(predict_full(model, fields, device), fields[..., 0])


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OperatorConfig(
        d_model=64,
        n_heads=4,
        encoder_layers=3,
        decoder_layers=2,
        token_count=32,
        conv_channels=32,
        x_bands=16,
        t_bands=8,
        dropout=0.1,
        dynamics="latent_pde",
        latent_channels=96,
        latent_length=64,
        rollout_steps=9,
        pde_substeps=1,
        pde_modes=32,
        pde_layers=1,
        pde_rank=24,
        output_mode="grid",
        grid_hidden=48,
        grid_blocks=3,
        grid_length=64,
        grid_block="dense",
    )
    model = CoordAttentionOperator(cfg).to(device)
    for projection in model.grid_decoder.skip_projs:
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )
    print(f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print("missing keys:", missing, flush=True)

    train = load_split(DATA / "burgers_nu0.01_train_000.parquet", 8000).to(device)
    dev = load_split(DATA / "burgers_nu0.01_dev.parquet", 1000).to(device)
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(device)
    model.eval()
    initial = {
        "dev": evaluate(model, dev, device),
        "test": evaluate(model, test, device),
    }
    print(f"initial {initial}", flush=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    skip_parameters = list(model.grid_decoder.skip_projs.parameters())
    skip_parameters += list(model.grid_decoder.skip_gate.parameters())
    for parameter in skip_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(skip_parameters, lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STEPS,
        eta_min=1e-5,
    )
    best_dev = initial["dev"]
    best_state = copy.deepcopy(model.state_dict())
    for step in range(STEPS):
        index = torch.randint(0, train.size(0), (BATCH,), device=device)
        target = train[index, :, :, 0]
        prediction = model.forward_grid(target[:, :, 0])
        loss = torch.mean((prediction - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skip_parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 200 == 0:
            dev_l2 = evaluate(model, dev, device)
            print(
                f"skip step {step + 1}: loss={loss.item():.5f} "
                f"dev={dev_l2:.5f}",
                flush=True,
            )
            if dev_l2 < best_dev:
                best_dev = dev_l2
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    skipped = {
        "dev": evaluate(model, dev, device),
        "test": evaluate(model, test, device),
    }
    print(f"skip-trained {skipped}", flush=True)

    diagnostics = []
    block_state = copy.deepcopy(model.state_dict())
    for block_index in range(len(model.grid_decoder.blocks)):
        model.load_state_dict(block_state)
        final_conv = model.grid_decoder.blocks[block_index].body[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        diagnostics.append(
            {
                "removed_block": block_index,
                "dev": evaluate(model, dev, device),
                "test": evaluate(model, test, device),
            }
        )
    model.load_state_dict(block_state)
    print("block diagnostics:", diagnostics, flush=True)

    chosen = min(diagnostics, key=lambda row: row["dev"])
    corrected = {
        "dev": skipped["dev"],
        "test": skipped["test"],
        "corrected_block": None,
    }
    if chosen["dev"] < skipped["dev"]:
        final_conv = model.grid_decoder.blocks[
            chosen["removed_block"]
        ].body[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        corrected.update(
            {
                "dev": chosen["dev"],
                "test": chosen["test"],
                "corrected_block": chosen["removed_block"],
            }
        )
    print(f"corrected {corrected}", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "operator": cfg.__dict__},
        OUTPUT / "model.pt",
    )
    (OUTPUT / "diagnostics.json").write_text(
        json.dumps(
            {
                "initial": initial,
                "skip_trained": skipped,
                "block_ablation": diagnostics,
                "corrected": corrected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
