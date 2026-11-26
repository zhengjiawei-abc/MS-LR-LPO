"""Shared normalized PDE residuals for 1D operator learning."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from burgers_common import NU, T_FINAL


DIFFUSION_SORPTION_OFFICIAL = {
    "D": 5.0e-4,
    "porosity": 0.29,
    "rho_s": 2880.0,
    "k_f": 3.5e-4,
    "n_f": 0.874,
    "solubility": 1.0,
    "t_final": 500.0,
}


def pde_derivative_fields(
    fields: torch.Tensor,
    equation: str,
    beta: float,
    diffusion_sorption_params: dict[str, float] | None = None,
    nu: float = NU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual, u_t, u_x, and u_xx for [B, X, T] fields."""
    params = dict(DIFFUSION_SORPTION_OFFICIAL)
    if diffusion_sorption_params:
        params.update(diffusion_sorption_params)
    dx = 1.0 / fields.size(1)
    dt = (
        float(params["t_final"])
        if equation == "diffusion_sorption"
        else T_FINAL
    ) / max(fields.size(2) - 1, 1)
    time_padded = F.pad(fields, (1, 1), mode="replicate")
    u_t = (time_padded[:, :, 2:] - time_padded[:, :, :-2]) / (2 * dt)
    u_x = (torch.roll(fields, -1, 1) - torch.roll(fields, 1, 1)) / (2 * dx)
    if equation == "burgers":
        u_xx = (
            torch.roll(fields, -1, 1)
            - 2.0 * fields
            + torch.roll(fields, 1, 1)
        ) / (dx * dx)
        residual = u_t + fields * u_x - nu * u_xx
    elif equation == "advection":
        u_xx = torch.zeros_like(fields)
        residual = u_t + beta * u_x
    elif equation == "diffusion_sorption":
        u_xx = (
            torch.roll(fields, -1, 1)
            - 2.0 * fields
            + torch.roll(fields, 1, 1)
        ) / (dx * dx)
        retardation = (
            1.0
            + ((1.0 - params["porosity"]) / params["porosity"])
            * params["rho_s"]
            * params["k_f"]
            * params["n_f"]
            * fields.clamp_min(1.0e-6) ** (params["n_f"] - 1.0)
        )
        residual = retardation * u_t - params["D"] * u_xx
    else:
        raise ValueError(f"unsupported equation: {equation}")
    return residual, u_t, u_x, u_xx


def normalized_pde_feature_channels(
    fields: torch.Tensor,
    equation: str,
    beta: float,
    diffusion_sorption_params: dict[str, float] | None = None,
    nu: float = NU,
) -> torch.Tensor:
    """Build scale-normalized PDE features for [B, X, T] fields."""
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        beta,
        diffusion_sorption_params,
        nu,
    )

    if equation == "diffusion_sorption":
        params = dict(DIFFUSION_SORPTION_OFFICIAL)
        if diffusion_sorption_params:
            params.update(diffusion_sorption_params)
        retardation = (
            1.0
            + ((1.0 - params["porosity"]) / params["porosity"])
            * params["rho_s"]
            * params["k_f"]
            * params["n_f"]
            * fields.clamp_min(1.0e-6) ** (params["n_f"] - 1.0)
        )
        diffusion = params["D"] * u_xx
        scale = (
            (retardation * u_t).square() + diffusion.square()
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

    if equation == "burgers":
        scale = (
            u_t.square()
            + (fields * u_x).square()
            + (nu * u_xx).square()
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


def normalized_diffusion_sorption_residual(
    fields: torch.Tensor,
    parameters: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freundlich diffusion-sorption residual for [B, X, T] fields.

    The official finite-volume solver applies a left Dirichlet source and a
    right flux boundary. We therefore score the physical interior only.
    """
    params = dict(DIFFUSION_SORPTION_OFFICIAL)
    if parameters:
        params.update(parameters)
    interior = fields[:, 1:-1, 1:-1]
    dx = 1.0 / fields.size(1)
    dt = float(params["t_final"]) / max(fields.size(2) - 1, 1)
    concentration = interior
    c_t = (
        fields[:, 1:-1, 2:] - fields[:, 1:-1, :-2]
    ) / (2.0 * dt)
    c_xx = (
        fields[:, 2:, 1:-1]
        - 2.0 * fields[:, 1:-1, 1:-1]
        + fields[:, :-2, 1:-1]
    ) / (dx * dx)
    retardation = (
        1.0
        + ((1.0 - params["porosity"]) / params["porosity"])
        * params["rho_s"]
        * params["k_f"]
        * params["n_f"]
        * concentration.clamp_min(1.0e-6) ** (params["n_f"] - 1.0)
    )
    diffusion = params["D"] * c_xx
    residual = retardation * c_t - diffusion
    numerator = residual.square().mean(dim=(1, 2))
    scale = (
        (retardation * c_t).square() + diffusion.square()
    ).mean(dim=(1, 2)).clamp_min(1.0e-12)
    relative = (numerator / scale).mean()
    return relative, relative.sqrt()


def normalized_pde_residual(
    fields: torch.Tensor,
    equation: str,
    beta: float,
    diffusion_sorption_params: dict[str, float] | None = None,
    nu: float = NU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized residual MSE and RMS for [B, X, T] fields."""
    if equation == "diffusion_sorption":
        return normalized_diffusion_sorption_residual(
            fields,
            diffusion_sorption_params,
        )
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        beta,
        diffusion_sorption_params,
        nu,
    )
    if equation == "burgers":
        convective = fields * u_x
        denominator = (
            u_t.square()
            + convective.square()
            + (nu * u_xx).square()
        )
    elif equation == "advection":
        convective = beta * u_x
        denominator = u_t.square() + convective.square()
    numerator = residual.square().mean(dim=(1, 2))
    scale = denominator.mean(dim=(1, 2)).clamp_min(1.0e-12)
    relative = (numerator / scale).mean()
    return relative, relative.sqrt()


def normalized_pde_residual_per_sample(
    fields: torch.Tensor,
    equation: str,
    beta: float,
    nu: float = NU,
) -> torch.Tensor:
    if equation == "diffusion_sorption":
        raise ValueError("use the diffusion-sorption helper directly")
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        beta,
        nu=nu,
    )
    if equation == "burgers":
        denominator = (
            u_t.square()
            + (fields * u_x).square()
            + (nu * u_xx).square()
        )
    elif equation == "advection":
        denominator = u_t.square() + (beta * u_x).square()
    else:
        raise ValueError(f"unsupported equation: {equation}")
    numerator = residual.square().mean(dim=(1, 2))
    scale = denominator.mean(dim=(1, 2)).clamp_min(1.0e-12)
    return (numerator / scale).sqrt()


def normalized_pde_residual_region(
    fields: torch.Tensor,
    equation: str,
    beta: float,
    time_margin: int = 2,
    nu: float = NU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Second-order residual over the time interior."""
    if equation == "diffusion_sorption":
        return normalized_diffusion_sorption_residual(fields)
    residual, u_t, u_x, u_xx = pde_derivative_fields(
        fields,
        equation,
        beta,
        nu=nu,
    )
    if equation == "burgers":
        denominator = (
            u_t.square() + (fields * u_x).square() + (nu * u_xx).square()
        )
    elif equation == "advection":
        denominator = u_t.square() + (beta * u_x).square()
    else:
        raise ValueError(f"unsupported equation: {equation}")
    time_slice = slice(time_margin, -time_margin)
    numerator = residual[:, :, time_slice].square().mean(dim=(1, 2))
    scale = (
        denominator[:, :, time_slice].mean(dim=(1, 2)).clamp_min(1.0e-12)
    )
    relative = (numerator / scale).mean()
    return relative, relative.sqrt()


def high_order_pde_residual_region(
    fields: torch.Tensor,
    equation: str,
    nu: float = NU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourth-order space-time residual over the non-boundary interior."""
    dx = 1.0 / fields.size(1)
    dt = T_FINAL / max(fields.size(2) - 1, 1)
    center = fields[:, 2:-2, 2:-2]
    u_t = (
        fields[:, 2:-2, :-4]
        - 8.0 * fields[:, 2:-2, 1:-3]
        + 8.0 * fields[:, 2:-2, 3:-1]
        - fields[:, 2:-2, 4:]
    ) / (12.0 * dt)
    u_x = (
        fields[:, 4:, 2:-2]
        - 8.0 * fields[:, 3:-1, 2:-2]
        + 8.0 * fields[:, 1:-3, 2:-2]
        - fields[:, :-4, 2:-2]
    ) / (12.0 * dx)
    u_xx = (
        -fields[:, 4:, 2:-2]
        + 16.0 * fields[:, 3:-1, 2:-2]
        - 30.0 * center
        + 16.0 * fields[:, 1:-3, 2:-2]
        - fields[:, :-4, 2:-2]
    ) / (12.0 * dx * dx)
    if equation == "burgers":
        residual = u_t + center * u_x - nu * u_xx
        denominator = (
            u_t.square()
            + (center * u_x).square()
            + (nu * u_xx).square()
        )
    elif equation == "advection":
        residual = u_t + u_x
        denominator = u_t.square() + u_x.square()
    else:
        raise ValueError(f"unsupported equation: {equation}")
    numerator = residual.square().mean(dim=(1, 2))
    scale = denominator.mean(dim=(1, 2)).clamp_min(1.0e-12)
    relative = (numerator / scale).mean()
    return relative, relative.sqrt()
