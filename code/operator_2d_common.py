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
    diffusion: tuple[float, float] = (1.0e-3, 5.0e-3),
    reaction: float = 5.0e-3,
) -> torch.Tensor:
    """PDEBench 2D diffusion-reaction residual.

    The official dynamics are
    u_t = Du lap(u) + u - u^3 - k - v
    v_t = Dv lap(v) + u - v.
    prediction is [B, 2, X, Y, T].
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
    ru = du_dt - diffusion[0] * lap_u - (u - u.pow(3) - reaction - v)
    rv = dv_dt - diffusion[1] * lap_v - (u - v)
    return torch.stack([ru, rv], dim=1)


def pde_residual_2d_relative(
    prediction: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    residual = pde_residual_2d(prediction, coordinates)
    u = prediction[:, 0]
    v = prediction[:, 1]
    x, y, t = coordinates
    dx = float((x[-1] - x[0]) / max(x.numel() - 1, 1))
    dy = float((y[-1] - y[0]) / max(y.numel() - 1, 1))
    dt = float((t[-1] - t[0]) / max(t.numel() - 1, 1))
    u_t = torch.gradient(u, spacing=(dx, dy, dt), dim=(1, 2, 3))[2]
    v_t = torch.gradient(v, spacing=(dx, dy, dt), dim=(1, 2, 3))[2]
    scale = (
        u_t.square()
        + v_t.square()
        + (u - u.pow(3) - 5.0e-3 - v).square()
        + (u - v).square()
    ).flatten(1).mean(dim=1).clamp_min(1.0e-12)
    return residual.square().flatten(1).mean(dim=1) / scale


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


def ns_momentum_divergence_relative(
    prediction: torch.Tensor,
    coordinates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    force: torch.Tensor | None = None,
    particles: torch.Tensor | None = None,
    viscosity: float = 0.01,
) -> torch.Tensor:
    """Divergence of the NS momentum equation after eliminating pressure.

    prediction and force are [B, 2, X, Y, T] and particles is [B, X, Y, T].
    Interior finite differences are used because the official solver applies
    non-periodic velocity boundary conditions.
    """
    if prediction.size(1) < 2:
        raise ValueError("NS residual expects u/v channels")
    x, y, t = coordinates
    dx = float((x[-1] - x[0]) / max(x.numel() - 1, 1))
    dy = float((y[-1] - y[0]) / max(y.numel() - 1, 1))
    dt = float((t[-1] - t[0]) / max(t.numel() - 1, 1))
    u = prediction[:, :, 1:-1, 1:-1, 1:-1]
    v = prediction[:, :, 1:-1, 1:-1, 1:-1]
    center = prediction[:, :, 1:-1, 1:-1, 1:-1]
    u_center = center[:, 0]
    v_center = center[:, 1]
    u_t = (
        prediction[:, 0, 1:-1, 1:-1, 2:]
        - prediction[:, 0, 1:-1, 1:-1, :-2]
    ) / (2.0 * dt)
    v_t = (
        prediction[:, 1, 1:-1, 1:-1, 2:]
        - prediction[:, 1, 1:-1, 1:-1, :-2]
    ) / (2.0 * dt)
    u_x = (
        prediction[:, 0, 2:, 1:-1, 1:-1]
        - prediction[:, 0, :-2, 1:-1, 1:-1]
    ) / (2.0 * dx)
    u_y = (
        prediction[:, 0, 1:-1, 2:, 1:-1]
        - prediction[:, 0, 1:-1, :-2, 1:-1]
    ) / (2.0 * dy)
    v_x = (
        prediction[:, 1, 2:, 1:-1, 1:-1]
        - prediction[:, 1, :-2, 1:-1, 1:-1]
    ) / (2.0 * dx)
    v_y = (
        prediction[:, 1, 1:-1, 2:, 1:-1]
        - prediction[:, 1, 1:-1, :-2, 1:-1]
    ) / (2.0 * dy)
    lap_u = (
        prediction[:, 0, 2:, 1:-1, 1:-1]
        - 2.0 * prediction[:, 0, 1:-1, 1:-1, 1:-1]
        + prediction[:, 0, :-2, 1:-1, 1:-1]
    ) / (dx * dx) + (
        prediction[:, 0, 1:-1, 2:, 1:-1]
        - 2.0 * prediction[:, 0, 1:-1, 1:-1, 1:-1]
        + prediction[:, 0, 1:-1, :-2, 1:-1]
    ) / (dy * dy)
    lap_v = (
        prediction[:, 1, 2:, 1:-1, 1:-1]
        - 2.0 * prediction[:, 1, 1:-1, 1:-1, 1:-1]
        + prediction[:, 1, :-2, 1:-1, 1:-1]
    ) / (dx * dx) + (
        prediction[:, 1, 1:-1, 2:, 1:-1]
        - 2.0 * prediction[:, 1, 1:-1, 1:-1, 1:-1]
        + prediction[:, 1, 1:-1, :-2, 1:-1]
    ) / (dy * dy)
    if force is None:
        force_x = torch.zeros_like(u_t)
        force_y = torch.zeros_like(v_t)
    else:
        if force.dim() == 4:
            force = force.unsqueeze(-1).expand(-1, -1, -1, -1, u_t.size(-1))
        force_x = force[:, 0, 1:-1, 1:-1, :]
        force_y = force[:, 1, 1:-1, 1:-1, :]
        if particles is not None:
            rho = particles[:, 1:-1, 1:-1, 1:-1]
            force_x = rho * force_x
            force_y = rho * force_y
    momentum_x = u_t + u_center * u_x + v_center * u_y - viscosity * lap_u - force_x
    momentum_y = v_t + u_center * v_x + v_center * v_y - viscosity * lap_v - force_y
    dmx_dx = (momentum_x[:, 2:] - momentum_x[:, :-2]) / (2.0 * dx)
    dmy_dy = (momentum_y[:, :, 2:] - momentum_y[:, :, :-2]) / (2.0 * dy)
    dmx_dx = dmx_dx[:, 1:-1, 2:-2]
    dmy_dy = dmy_dy[:, 2:-2, 1:-1]
    divergence = dmx_dx + dmy_dy
    scale = (dmx_dx.square() + dmy_dy.square()).flatten(1).mean(
        dim=1
    ).clamp_min(1.0e-12)
    return divergence.square().flatten(1).mean(dim=1) / scale
