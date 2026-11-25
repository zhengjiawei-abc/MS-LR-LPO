"""Shared normalized PDE residuals for 1D operator learning."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from burgers_common import NU, T_FINAL


def normalized_pde_residual(
    fields: torch.Tensor,
    equation: str,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized residual MSE and RMS for [B, X, T] fields."""
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
        convective = fields * u_x
        residual = u_t + convective - NU * u_xx
        denominator = (
            u_t.square()
            + convective.square()
            + (NU * u_xx).square()
        )
    elif equation == "advection":
        convective = beta * u_x
        residual = u_t + convective
        denominator = u_t.square() + convective.square()
    else:
        raise ValueError(f"unsupported equation: {equation}")
    numerator = residual.square().mean(dim=(1, 2))
    scale = denominator.mean(dim=(1, 2)).clamp_min(1.0e-12)
    relative = (numerator / scale).mean()
    return relative, relative.sqrt()
