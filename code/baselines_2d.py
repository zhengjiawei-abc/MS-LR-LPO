"""Parameter-matched 2D Fourier neural operator baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class FNO2DConfig:
    field_channels: int = 2
    input_channels: int = 0
    width: int = 24
    modes_x: int = 8
    modes_y: int = 8
    layers: int = 4


@dataclass
class DeepONet2DConfig:
    field_channels: int = 2
    input_channels: int = 0
    basis: int = 64
    branch_hidden: int = 64
    trunk_hidden: int = 128
    input_grid: int = 32
    trunk_bands: int = 8


class SpectralConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y)
        self.weight_real = nn.Parameter(torch.randn(shape) * scale)
        self.weight_imag = nn.Parameter(torch.randn(shape) * scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = state.shape
        input_dtype = state.dtype
        if state.dtype in {torch.float16, torch.bfloat16}:
            state = state.float()
        spectrum = torch.fft.rfft2(state, norm="ortho")
        modes_x = min(self.modes_x, height // 2 + 1)
        modes_y = min(self.modes_y, width // 2 + 1)
        weight = torch.complex(self.weight_real, self.weight_imag)
        output = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        output[:, :, :modes_x, :modes_y] = torch.einsum(
            "bixy,ioxy->boxy",
            spectrum[:, :, :modes_x, :modes_y],
            weight[:, :, :modes_x, :modes_y],
        )
        output = torch.fft.irfft2(
            output,
            s=(height, width),
            norm="ortho",
        )
        return output.to(input_dtype)


class FNOBlock2D(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2D(width, width, modes_x, modes_y)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(state) + self.local(state))


class FNO2D(nn.Module):
    """Maps [B, C, X, Y] initial fields to [B, C, X, Y, T]."""

    def __init__(
        self,
        cfg: FNO2DConfig,
        nx: int,
        ny: int,
        nt: int,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.field_channels = cfg.field_channels
        self.lift = nn.Conv2d(
            cfg.input_channels or cfg.field_channels,
            cfg.width,
            1,
        )
        self.blocks = nn.ModuleList(
            FNOBlock2D(cfg.width, cfg.modes_x, cfg.modes_y)
            for _ in range(max(1, cfg.layers))
        )
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.width, cfg.width * 2, 1),
            nn.GELU(),
            nn.Conv2d(
                cfg.width * 2,
                cfg.field_channels * nt,
                1,
            ),
        )

    def forward_grid(self, u0: torch.Tensor) -> torch.Tensor:
        state = self.lift(u0)
        for block in self.blocks:
            state = block(state)
        output = self.proj(state)
        output = output.view(
            u0.size(0),
            self.field_channels,
            self.nt,
            self.nx,
            self.ny,
        )
        return output.permute(0, 1, 3, 4, 2)


class DeepONet2D(nn.Module):
    """Grid branch plus space-time trunk for 2D operator learning."""

    def __init__(
        self,
        cfg: DeepONet2DConfig,
        nx: int,
        ny: int,
        nt: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        t_min: float,
        t_max: float,
    ) -> None:
        super().__init__()
        input_channels = cfg.input_channels or cfg.field_channels
        flattened = input_channels * cfg.input_grid * cfg.input_grid
        self.field_channels = cfg.field_channels
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.input_grid = cfg.input_grid
        self.branch = nn.Sequential(
            nn.Linear(flattened, cfg.branch_hidden),
            nn.GELU(),
            nn.Linear(
                cfg.branch_hidden,
                cfg.field_channels * cfg.basis,
            ),
        )
        x = torch.linspace(x_min, x_max, nx)
        y = torch.linspace(y_min, y_max, ny)
        t = torch.linspace(t_min, t_max, nt)
        grid_x, grid_y, grid_t = torch.meshgrid(x, y, t, indexing="ij")
        coordinates = torch.stack(
            [
                (grid_x.flatten() - x_min) / max(x_max - x_min, 1.0e-8) * 2.0
                - 1.0,
                (grid_y.flatten() - y_min) / max(y_max - y_min, 1.0e-8) * 2.0
                - 1.0,
                (grid_t.flatten() - t_min) / max(t_max - t_min, 1.0e-8),
            ],
            dim=-1,
        )
        feature_parts = [coordinates]
        frequencies = math.pi * torch.arange(
            1,
            cfg.trunk_bands + 1,
            dtype=torch.float32,
        )
        phase = coordinates[:, :, None] * frequencies[None, None, :]
        feature_parts.extend([phase.sin().flatten(1), phase.cos().flatten(1)])
        trunk_features = torch.cat(feature_parts, dim=-1)
        self.register_buffer("coordinates", trunk_features)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_features.size(1), cfg.trunk_hidden),
            nn.GELU(),
            nn.Linear(cfg.trunk_hidden, cfg.basis),
        )
        self.bias = nn.Parameter(torch.zeros(1, cfg.field_channels, 1))

    def forward_grid(self, u0: torch.Tensor) -> torch.Tensor:
        coarse = F.adaptive_avg_pool2d(
            u0,
            (self.input_grid, self.input_grid),
        )
        branch = self.branch(coarse.flatten(1)).view(
            u0.size(0),
            self.field_channels,
            -1,
        )
        trunk = self.trunk(self.coordinates)
        output = torch.einsum("bcp,qp->bcq", branch, trunk) + self.bias
        return output.view(
            u0.size(0),
            self.field_channels,
            self.nx,
            self.ny,
            self.nt,
        )
