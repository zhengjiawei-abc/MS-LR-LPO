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
