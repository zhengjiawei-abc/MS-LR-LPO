"""PDEBench 1D Burgers (Nu0.01) 的公共工具：网格常数、数据加载、随机种子、误差。

网格约定经官方 PDEBench parquet 数据核对：
  u_t + u u_x = NU u_xx,  x ∈ [0, 1) 周期,  t ∈ [0, 2],  NU = 0.01 / π
parquet 是原始 1024 × 201 网格降采样后的 256 × 101：x_i = i / 256，t_j = 2 j / 100。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

X_LENGTH = 1.0
T_FINAL = 2.0
NU = 0.01 / math.pi
NX, NT = 256, 101


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def configure_backend(device: torch.device) -> None:
    """TF32 矩阵乘与 cuDNN 自动调优（fp32 存储，精度影响可忽略）。"""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_split(path: Path, max_samples: int) -> torch.Tensor:
    """读取 parquet 的前 max_samples 条，返回 [N, nx, nt, 1] float32。"""
    parquet = pq.ParquetFile(path)
    samples: List[np.ndarray] = []
    for batch in parquet.iter_batches(batch_size=32, columns=["tensor"]):
        column = batch.column(0)
        for index in range(batch.num_rows):
            if len(samples) >= max_samples:
                break
            samples.append(np.asarray(column[index].as_py(), dtype=np.float32))
        if len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError(f"No samples loaded from {path}")
    return torch.from_numpy(np.stack(samples, axis=0))


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """逐样本相对 L2 的平均。两者形状为 [N, ...]。"""
    error = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    scale = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1.0e-8)
    return float((error / scale).mean())


def grid_coordinates(device: torch.device, nx: int = NX, nt: int = NT):
    """整场查询坐标，按 [nx, nt] 行优先展平，返回 x, t 各 [nx * nt, 1]（物理单位）。"""
    x = torch.arange(nx, device=device, dtype=torch.float32) * (X_LENGTH / nx)
    t = torch.linspace(0.0, T_FINAL, nt, device=device)
    return x.repeat_interleave(nt).unsqueeze(1), t.repeat(nx).unsqueeze(1)
