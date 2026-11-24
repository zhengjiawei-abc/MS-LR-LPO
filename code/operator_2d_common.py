"""Shared utilities for the 2D PDE operator experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Grid2DSpec:
    nx: int
    ny: int
    nt: int
    channels: int
    input_channels: int = 0
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @property
    def x_length(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_length(self) -> float:
        return self.y_max - self.y_min

    @property
    def t_length(self) -> float:
        return self.t_max - self.t_min

    @property
    def effective_input_channels(self) -> int:
        return self.input_channels or self.channels


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    scale = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1.0e-8)
    return float((error / scale).mean())


def load_npz_split(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        inputs = torch.from_numpy(np.asarray(data["inputs"], dtype=np.float32))
        targets = torch.from_numpy(np.asarray(data["targets"], dtype=np.float32))
    return inputs, targets


def load_spec(path: Path) -> Grid2DSpec:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    channels = metadata["channels"]
    return Grid2DSpec(
        nx=int(metadata["nx"]),
        ny=int(metadata["ny"]),
        nt=int(metadata["nt"]),
        channels=len(channels) if isinstance(channels, list) else int(channels),
        input_channels=int(metadata.get("input_channels", 0)),
        x_min=float(metadata["x_min"]),
        x_max=float(metadata["x_max"]),
        y_min=float(metadata["y_min"]),
        y_max=float(metadata["y_max"]),
        t_min=float(metadata["t_min"]),
        t_max=float(metadata["t_max"]),
    )


def space_time_coordinates(
    spec: Grid2DSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(spec.x_min, spec.x_max, spec.nx, device=device)
    y = torch.linspace(spec.y_min, spec.y_max, spec.ny, device=device)
    t = torch.linspace(spec.t_min, spec.t_max, spec.nt, device=device)
    return x, y, t


def pde_residual_2d(
    prediction: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    diffusion: tuple[float, float] = (1.0e-3, 5.0e-4),
    reaction: tuple[float, float, float] = (0.04, 0.06, 0.0),
) -> torch.Tensor:
    """Finite-difference Gray-Scott residual for two-channel predictions.

    prediction is [B, 2, X, Y, T]. The residual is diagnostic because the
    public dataset does not expose per-sample reaction parameters.
    """
    if prediction.size(1) != 2:
        raise ValueError("Gray-Scott residual expects exactly two channels")
    x, y, t = coordinates
    dx = float(x[1] - x[0]) if x.numel() > 1 else 1.0
    dy = float(y[1] - y[0]) if y.numel() > 1 else 1.0
    dt = float(t[1] - t[0]) if t.numel() > 1 else 1.0
    d_x = (x[-1] - x[0]) / max(x.numel() - 1, 1)
    d_y = (y[-1] - y[0]) / max(y.numel() - 1, 1)
    dx = float(d_x) if dx == 0.0 else dx
    dy = float(d_y) if dy == 0.0 else dy
    dt = float((t[-1] - t[0]) / max(t.numel() - 1, 1))

    u = prediction[:, 0]
    v = prediction[:, 1]
    du_dt = torch.gradient(u, spacing=(dx, dy, dt), dim=(1, 2, 3))[2]
    dv_dt = torch.gradient(v, spacing=(dx, dy, dt), dim=(1, 2, 3))[2]
    lap_u = (
        torch.gradient(
            torch.gradient(u, spacing=(dx, dy, dt), dim=(1, 2, 3))[0],
            spacing=(dx, dy, dt),
            dim=(1, 2, 3),
        )[0]
        + torch.gradient(
            torch.gradient(u, spacing=(dx, dy, dt), dim=(1, 2, 3))[1],
            spacing=(dx, dy, dt),
            dim=(1, 2, 3),
        )[1]
    )
    lap_v = (
        torch.gradient(
            torch.gradient(v, spacing=(dx, dy, dt), dim=(1, 2, 3))[0],
            spacing=(dx, dy, dt),
            dim=(1, 2, 3),
        )[0]
        + torch.gradient(
            torch.gradient(v, spacing=(dx, dy, dt), dim=(1, 2, 3))[1],
            spacing=(dx, dy, dt),
            dim=(1, 2, 3),
        )[1]
    )
    feed, kill, _ = reaction
    ru = du_dt - diffusion[0] * lap_u - (-u * v.square() + feed * (1.0 - u))
    rv = dv_dt - diffusion[1] * lap_v - (u * v.square() - (feed + kill) * v)
    return torch.stack([ru, rv], dim=1)


def incompressible_divergence(
    prediction: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Periodic divergence of a velocity prediction [B, 2, X, Y, T]."""
    if prediction.size(1) < 2:
        raise ValueError("incompressible divergence expects at least (u, v)")
    x, y, _ = coordinates
    dx = float(x[1] - x[0]) if x.numel() > 1 else 1.0
    dy = float(y[1] - y[0]) if y.numel() > 1 else 1.0
    u = prediction[:, 0]
    v = prediction[:, 1]
    du_dx = torch.gradient(u, spacing=dx, dim=1)[0]
    dv_dy = torch.gradient(v, spacing=dy, dim=2)[0]
    return du_dx + dv_dy


def divergence_relative_per_sample(
    prediction: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    divergence = incompressible_divergence(prediction, coordinates)
    x, y, _ = coordinates
    dx = float(x[1] - x[0]) if x.numel() > 1 else 1.0
    dy = float(y[1] - y[0]) if y.numel() > 1 else 1.0
    u = prediction[:, 0]
    v = prediction[:, 1]
    du_dx = torch.gradient(u, spacing=dx, dim=1)[0]
    dv_dy = torch.gradient(v, spacing=dy, dim=2)[0]
    scale = (
        du_dx.square() + dv_dy.square()
    ).flatten(1).mean(dim=1).clamp_min(1.0e-12)
    return divergence.flatten(1).square().mean(dim=1) / scale


def divergence_relative_loss(
    prediction: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    return divergence_relative_per_sample(prediction, coordinates).mean()
