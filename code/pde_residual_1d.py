"""Shared normalized PDE residuals for 1D operator learning."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from burgers_common import NU, T_FINAL


def pde_derivative_fields(
    fields: torch.Tensor,
    equation: str,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual, u_t, u_x, and u_xx for [B, X, T] fields."""
    dx = 1.0 / fields.size(1)
    dt = T_FINAL / max(fields.size(2) - 1, 1)
    time_padded = F.pad(fields, (1, 1), mode="replicate")
    u_t = (time_padded[:, :, 2:] - time_padded[:, :, :-2]) / (2 * dt)
    u_x = (torch.roll(fields, -1, 1) - torch.roll(fields, 1, 1)) / (2 * dx)
    if equation == "burgers":
        u_xx = (
            torch.roll(fields, -1, 1)
            - 2.0 * fields
            + torch.roll(fields, 1, 1)
        ) / (dx * dx)
        residual = u_t + fields * u_x - NU * u_xx
    elif equation == "advection":
        u_xx = torch.zeros_like(fields)
        residual = u_t + beta * u_x
    else:
        raise ValueError(f"unsupported equation: {equation}")
    return residual, u_t, u_x, u_xx


def normalized_pde_feature_channels(
    fields: torch.Tensor,
    equation: str,
    beta: float,
) -> torch.Tensor:
    """Build scale-normalized PDE features for [B, X, T] fields."""
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        beta,
    )
    if equation == "burgers":
        scale = (
            u_t.square()
            + (fields * u_x).square()
            + (NU * u_xx).square()
        ).mean(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    else:
        scale = (
            u_t.square() + (beta * u_x).square()
        ).mean(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    view = (-1, 1, 1)
    u_x_scale = u_x.square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    u_xx_scale = u_xx.square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-6)
    return torch.stack(
        (
            residual / scale.view(view),
            u_x / u_x_scale.view(view),
            u_xx / u_xx_scale.view(view),
        ),
        dim=1,
    )


def normalized_pde_residual(
    fields: torch.Tensor,
    equation: str,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized residual MSE and RMS for [B, X, T] fields."""
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        beta,
    )
    if equation == "burgers":
        convective = fields * u_x
        denominator = (
            u_t.square()
            + convective.square()
            + (NU * u_xx).square()
        )
    elif equation == "advection":
        convective = beta * u_x
        denominator = u_t.square() + convective.square()
    numerator = residual.square().mean(dim=(1, 2))
    scale = denominator.mean(dim=(1, 2)).clamp_min(1.0e-12)
    relative = (numerator / scale).mean()
    return relative, relative.sqrt()
