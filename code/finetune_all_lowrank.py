"""Compress all decoder blocks with SVD and recover from the best model."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from burgers_common import load_split, relative_l2
from coord_attention_operator import (
    CoordAttentionOperator,
    GridResBlock,
    LowRankGridResBlock,
    OperatorConfig,
)
from train_coord_attention import predict_full


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "pdebench_burgers_official"
RUNS = ROOT / "experiments" / "outputs_burgers_dataset" / "official_n8000"
SOURCE = RUNS / "latent_pde_best_skip_corrected" / "model.pt"
OUTPUT = RUNS / "latent_pde_all_lowrank"
STEPS = 5000
BATCH = 64
RANK = 16
DEVICE = torch.device("cuda")


def evaluate(model, fields):
    return relative_l2(predict_full(model, fields, DEVICE), fields[..., 0])


def factorize(first: nn.Conv2d, second: nn.Conv2d, block: LowRankGridResBlock) -> None:
    for source, depth_conv, point_conv in (
        (first, block.body[0], block.body[1]),
        (second, block.body[3], block.body[4]),
    ):
        weight = source.weight.detach().reshape(source.out_channels, -1)
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        rank = block.rank
        depth_conv.weight.data.copy_(
            vh[:rank].reshape(rank, source.in_channels, 3, 3)
        )
        point_conv.weight.data.copy_(
            (u[:, :rank] * s[:rank]).reshape(source.out_channels, rank, 1, 1)
        )
        point_conv.bias.data.copy_(source.bias)


def copy_block(source_state, index: int, target: LowRankGridResBlock) -> None:
    dense = GridResBlock(48, "dense").to(DEVICE)
    dense.norm.load_state_dict(
        {
            "weight": source_state[f"grid_decoder.blocks.{index}.norm.weight"],
            "bias": source_state[f"grid_decoder.blocks.{index}.norm.bias"],
        }
    )
    dense.body[0].load_state_dict(
        {
            "weight": source_state[f"grid_decoder.blocks.{index}.body.0.weight"],
            "bias": source_state[f"grid_decoder.blocks.{index}.body.0.bias"],
        }
    )
    dense.body[2].load_state_dict(
        {
            "weight": source_state[f"grid_decoder.blocks.{index}.body.2.weight"],
            "bias": source_state[f"grid_decoder.blocks.{index}.body.2.bias"],
        }
    )
    target.norm.load_state_dict(dense.norm.state_dict())
    factorize(dense.body[0], dense.body[2], target)


def main() -> None:
    checkpoint = torch.load(SOURCE, map_location=DEVICE, weights_only=False)
    operator = dict(checkpoint["operator"])
    operator.update(
        {
            "grid_lowrank_rank": RANK,
            "grid_last_rank": 0,
            "grid_last_depthwise": False,
        }
    )
    cfg = OperatorConfig(**operator)
    model = CoordAttentionOperator(cfg).to(DEVICE)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in checkpoint["state_dict"].items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    for index, block in enumerate(model.grid_decoder.blocks):
        copy_block(checkpoint["state_dict"], index, block)

    train = load_split(
        DATA / "burgers_nu0.01_train_000.parquet", 8000
    ).to(DEVICE)
    dev = load_split(DATA / "burgers_nu0.01_dev.parquet", 1000).to(DEVICE)
    test = load_split(DATA / "burgers_nu0.01_test.parquet", 1000).to(DEVICE)
    model.eval()
    initial = {"dev": evaluate(model, dev), "test": evaluate(model, test)}
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"initial={initial} params={parameter_count}", flush=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.grid_decoder.blocks.parameters())
    trainable += list(model.grid_decoder.skip_projs.parameters())
    trainable += list(model.grid_decoder.skip_gate.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STEPS,
        eta_min=1e-6,
    )
    best_dev = initial["dev"]
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    for step in range(STEPS):
        index = torch.randint(0, train.size(0), (BATCH,), device=DEVICE)
        target = train[index, :, :, 0]
        prediction = model.forward_grid(target[:, :, 0])
        loss = torch.mean((prediction - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 500 == 0:
            dev_l2 = evaluate(model, dev)
            print(
                f"step={step + 1} loss={loss.item():.6f} dev={dev_l2:.6f}",
                flush=True,
            )
            if dev_l2 < best_dev:
                best_dev = dev_l2
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final = {"dev": evaluate(model, dev), "test": evaluate(model, test)}
    result = {
        "parameter_count": parameter_count,
        "initial": initial,
        "final": final,
        "rank": RANK,
        "training_seconds": time.time() - started,
        "config": cfg.__dict__,
    }
    print(json.dumps(result, indent=2), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "operator": cfg.__dict__},
        OUTPUT / "model.pt",
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
