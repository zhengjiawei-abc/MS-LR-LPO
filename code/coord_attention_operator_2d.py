"""Multi-scale latent PDE operator for two-dimensional space-time fields."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Operator2DConfig:
    field_channels: int = 2
    input_channels: int = 0
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 2
    token_grid: int = 8
    conv_channels: int = 40
    spatial_bands: int = 6
    time_bands: int = 5
    dropout: float = 0.05
    latent_channels: int = 40
    latent_grid: int = 16
    latent_recon: bool = False
    rollout_steps: int = 17
    pde_substeps: int = 1
    pde_modes_x: int = 8
    pde_modes_y: int = 8
    pde_layers: int = 1
    pde_rank: int = 8
    grid_hidden: int = 64
    grid_length_x: int = 32
    grid_length_y: int = 32
    spatial_blocks: int = 2
    temporal_blocks: int = 2
    spatial_rank: int = 16
    temporal_kernel: int = 5


class ContextEncoder2D(nn.Module):
    """Initial field [B, C, X, Y] -> local features and global tokens."""

    def __init__(self, cfg: Operator2DConfig) -> None:
        super().__init__()
        c = cfg.conv_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                cfg.input_channels or cfg.field_channels,
                c,
                5,
                padding=2,
                padding_mode="circular",
            ),
            nn.SiLU(),
            nn.Conv2d(
                c,
                c,
                5,
                padding=2,
                padding_mode="circular",
            ),
            nn.SiLU(),
        )
        self.token_grid = cfg.token_grid
        self.proj = nn.Linear(c, cfg.d_model)
        self.pos = nn.Parameter(
            torch.zeros(1, cfg.token_grid * cfg.token_grid, cfg.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            2 * cfg.d_model,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            cfg.encoder_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward_with_features(self, u0: torch.Tensor):
        local = self.stem(u0)
        pooled = F.adaptive_avg_pool2d(
            local, (self.token_grid, self.token_grid)
        )
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens) + self.pos
        return self.norm(self.encoder(tokens)), local

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(u0)[0]


class LatentPDEEncoder2D(nn.Module):
    """Initial field -> periodic latent grid [B, C, H, W]."""

    def __init__(self, cfg: Operator2DConfig) -> None:
        super().__init__()
        c = cfg.latent_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                cfg.input_channels or cfg.field_channels,
                c,
                5,
                padding=2,
                padding_mode="circular",
            ),
            nn.SiLU(),
            nn.Conv2d(
                c,
                c,
                5,
                padding=2,
                padding_mode="circular",
            ),
            nn.SiLU(),
        )
        self.latent_grid = cfg.latent_grid
        self.field_channels = cfg.field_channels
        self.decode = (
            nn.Sequential(
                nn.Conv2d(
                    c,
                    c,
                    3,
                    padding=1,
                    padding_mode="circular",
                ),
                nn.SiLU(),
                nn.Conv2d(
                    c,
                    cfg.field_channels,
                    3,
                    padding=1,
                    padding_mode="circular",
                ),
            )
            if cfg.latent_recon
            else None
        )

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        features = self.stem(u0)
        return F.adaptive_avg_pool2d(
            features, (self.latent_grid, self.latent_grid)
        )

    def reconstruct(
        self,
        u0: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if self.decode is None:
            raise ValueError("latent reconstruction head is disabled")
        restored = self.decode(self.forward(u0))
        return F.interpolate(
            restored,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class LatentSpectralMixer2D(nn.Module):
    """Low-rank complex mixing across latent channels and 2D Fourier modes."""

    def __init__(
        self,
        channels: int,
        modes_x: int,
        modes_y: int,
        rank: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.modes_x = min(max(1, modes_x), height // 2 + 1)
        self.modes_y = min(max(1, modes_y), width // 2 + 1)
        self.rank = max(1, min(rank, channels))
        self.to_modes = nn.Conv2d(channels, self.rank, 1)
        self.out_mix = nn.Conv2d(channels, channels, 1)
        shape = (self.rank, channels, self.modes_x, self.modes_y)
        self.weight_real = nn.Parameter(torch.randn(shape) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(shape) * 0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        input_dtype = state.dtype
        if state.dtype in {torch.float16, torch.bfloat16}:
            state = state.float()
        with torch.autocast(device_type=state.device.type, enabled=False):
            spectrum = torch.fft.rfft2(
                self.to_modes(state),
                norm="ortho",
            )
            cropped = spectrum[..., : self.modes_x, : self.modes_y]
            weight = torch.complex(self.weight_real, self.weight_imag)
            mixed = torch.einsum("brxy,rcyx->bcxy", cropped, weight)
            full = torch.zeros(
                mixed.shape[0],
                mixed.shape[1],
                self.height,
                self.width // 2 + 1,
                dtype=mixed.dtype,
                device=mixed.device,
            )
            full[..., : self.modes_x, : self.modes_y] = mixed
            output = self.out_mix(torch.fft.irfft2(
                full,
                s=(self.height, self.width),
                norm="ortho",
            ))
        return output.to(input_dtype)


class LatentPDEBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_x: int,
        modes_y: int,
        rank: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                padding_mode="circular",
            ),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                padding_mode="circular",
            ),
        )
        self.spectral = LatentSpectralMixer2D(
            channels,
            modes_x,
            modes_y,
            rank,
            height,
            width,
        )
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        update = F.silu(self.local(self.norm(state)) + self.spectral(state))
        return self.proj(update)


class LatentPseudoSpectralPDE2D(nn.Module):
    """Advection-diffusion latent evolution with learned local reactions."""

    def __init__(self, cfg: Operator2DConfig) -> None:
        super().__init__()
        channels = cfg.latent_channels
        height = cfg.latent_grid
        width = cfg.latent_grid
        self.height = height
        self.width = width
        self.rollout_steps = max(2, cfg.rollout_steps)
        self.substeps = max(1, cfg.pde_substeps)
        self.modes_x = min(max(1, cfg.pde_modes_x), height // 2 + 1)
        self.modes_y = min(max(1, cfg.pde_modes_y), width // 2 + 1)
        self.encoder = LatentPDEEncoder2D(cfg)
        self.nonlinear = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            padding_mode="circular",
        )
        self.reaction = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.reaction.weight)
        nn.init.zeros_(self.reaction.bias)
        self.blocks = nn.ModuleList(
            LatentPDEBlock2D(
                channels,
                self.modes_x,
                self.modes_y,
                cfg.pde_rank,
                height,
                width,
            )
            for _ in range(max(0, cfg.pde_layers))
        )
        self.log_advection_x = nn.Parameter(torch.zeros(channels))
        self.log_advection_y = nn.Parameter(torch.zeros(channels))
        self.log_diffusion = nn.Parameter(
            torch.full((channels,), math.log(math.expm1(0.01)))
        )
        kx = 2.0 * math.pi * torch.fft.fftfreq(height)
        ky = 2.0 * math.pi * torch.fft.rfftfreq(width)
        self.register_buffer("kx", kx[None, None, :, None])
        self.register_buffer("ky", ky[None, None, None, :])
        mask = torch.zeros(height, width // 2 + 1)
        mask[: self.modes_x, : self.modes_y] = 1.0
        self.register_buffer("mask", mask)

    def _linear_step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        input_dtype = state.dtype
        if state.dtype in {torch.float16, torch.bfloat16}:
            state = state.float()
        velocity_x = 0.5 * torch.tanh(self.log_advection_x)
        velocity_y = 0.5 * torch.tanh(self.log_advection_y)
        diffusion = F.softplus(self.log_diffusion)
        linear = (
            -1j
            * (
                self.kx * velocity_x[None, :, None, None]
                + self.ky * velocity_y[None, :, None, None]
            )
            - diffusion[None, :, None, None]
            * (self.kx.square() + self.ky.square())
        )
        spectrum = torch.fft.rfft2(state, norm="ortho") * self.mask
        advanced = spectrum * torch.exp(linear * dt)
        output = torch.fft.irfft2(
            advanced,
            s=(self.height, self.width),
            norm="ortho",
        )
        return output.to(input_dtype)

    def _nonlinear(self, state: torch.Tensor) -> torch.Tensor:
        update = 0.1 * torch.tanh(self.nonlinear(state)) + self.reaction(state)
        for block in self.blocks:
            update = update + block(state)
        return update

    def _step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        state = self._linear_step(state, dt) + dt * self._nonlinear(state)
        input_dtype = state.dtype
        spectral_state = (
            state.float()
            if state.dtype in {torch.float16, torch.bfloat16}
            else state
        )
        spectrum = torch.fft.rfft2(
            spectral_state,
            norm="ortho",
        ) * self.mask
        output = torch.fft.irfft2(
            spectrum,
            s=(self.height, self.width),
            norm="ortho",
        )
        return output.to(input_dtype)

    def rollout(self, u0: torch.Tensor) -> torch.Tensor:
        """Return [B, C, H, W, K] latent trajectory."""
        state = self.encoder(u0)
        states = [state]
        dt = 1.0 / (self.rollout_steps - 1)
        step_dt = dt / self.substeps
        for _ in range(1, self.rollout_steps):
            for _ in range(self.substeps):
                state = self._step(state, step_dt)
            states.append(state)
        return torch.stack(states, dim=-1)


class LowRankSpatialResBlock2D(nn.Module):
    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        rank = max(1, min(rank, channels))
        self.norm = nn.GroupNorm(1, channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, rank, 3, padding=1, bias=False),
            nn.Conv2d(rank, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, rank, 3, padding=1, bias=False),
            nn.Conv2d(rank, channels, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.body(self.norm(state))


class TemporalResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.body = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel,
                padding=kernel // 2,
                groups=channels,
            ),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.body(self.norm(state))


class GridTrajectoryDecoder2D(nn.Module):
    """Factorized spatial-temporal decoder from a latent trajectory."""

    def __init__(
        self,
        cfg: Operator2DConfig,
        nx: int,
        ny: int,
        nt: int,
    ) -> None:
        super().__init__()
        channels = cfg.latent_channels
        hidden = cfg.grid_hidden
        self.hidden = hidden
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.grid_x = cfg.grid_length_x
        self.grid_y = cfg.grid_length_y
        self.context_proj = nn.Linear(cfg.d_model, channels)
        self.input_proj = nn.Conv2d(channels, hidden, 1)
        self.x_proj = nn.Linear(2 * cfg.spatial_bands, hidden)
        self.y_proj = nn.Linear(2 * cfg.spatial_bands, hidden)
        self.t_proj = nn.Linear(2 * cfg.time_bands + 1, hidden)
        self.spatial = nn.ModuleList(
            LowRankSpatialResBlock2D(hidden, cfg.spatial_rank)
            for _ in range(max(1, cfg.spatial_blocks))
        )
        self.temporal = nn.ModuleList(
            TemporalResBlock1D(hidden, cfg.temporal_kernel)
            for _ in range(max(1, cfg.temporal_blocks))
        )
        self.head = nn.Conv2d(hidden, cfg.field_channels, 1)
        self.skip_projs = nn.ModuleList(
            nn.Conv2d(cfg.conv_channels, cfg.field_channels, 1)
            for _ in range(3)
        )
        self.skip_gate = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * nt),
        )

        x = torch.linspace(0.0, 1.0, self.grid_x)
        y = torch.linspace(0.0, 1.0, self.grid_y)
        t = torch.linspace(0.0, 1.0, cfg.rollout_steps)
        x_freq = 2.0 * math.pi * torch.arange(1, cfg.spatial_bands + 1)
        y_freq = 2.0 * math.pi * torch.arange(1, cfg.spatial_bands + 1)
        t_freq = math.pi * 2.0 ** torch.arange(cfg.time_bands)
        x_phase = x[:, None] * x_freq[None, :]
        y_phase = y[:, None] * y_freq[None, :]
        t_phase = t[:, None] * t_freq[None, :]
        self.register_buffer(
            "x_features",
            torch.cat([x_phase.sin(), x_phase.cos()], dim=-1),
        )
        self.register_buffer(
            "y_features",
            torch.cat([y_phase.sin(), y_phase.cos()], dim=-1),
        )
        self.register_buffer(
            "t_features",
            torch.cat(
                [t_phase.sin(), t_phase.cos(), t[:, None]],
                dim=-1,
            ),
        )

    def forward(
        self,
        trajectory: torch.Tensor,
        context: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, _, _, rollout = trajectory.shape
        hidden = self.hidden
        context_grid = self.context_proj(context)
        context_grid = context_grid.transpose(1, 2).reshape(
            batch,
            channels,
            int(math.sqrt(context.size(1))),
            int(math.sqrt(context.size(1))),
        )
        context_grid = F.interpolate(
            context_grid,
            size=(self.grid_x, self.grid_y),
            mode="bilinear",
            align_corners=False,
        )
        state = F.interpolate(
            trajectory.permute(0, 1, 4, 2, 3),
            size=(rollout, self.grid_x, self.grid_y),
            mode="trilinear",
            align_corners=False,
        )
        state = state + context_grid.unsqueeze(2)
        state = state.permute(0, 1, 3, 4, 2)
        state2d = state.permute(0, 4, 1, 2, 3).reshape(
            batch * rollout,
            channels,
            self.grid_x,
            self.grid_y,
        )
        state = self.input_proj(state2d)

        x_pos = self.x_proj(self.x_features)
        y_pos = self.y_proj(self.y_features)
        space_pos = x_pos[:, None, :] + y_pos[None, :, :]
        time_pos = self.t_proj(self.t_features)
        position = (
            space_pos[:, :, None, :] + time_pos[None, None, :, :]
        ).permute(2, 3, 0, 1)
        position = position.unsqueeze(0).expand(
            batch, -1, -1, -1, -1
        ).reshape(batch * rollout, hidden, self.grid_x, self.grid_y)
        state = state + position
        for block in self.spatial:
            state = block(state)

        state = state.view(
            batch,
            rollout,
            hidden,
            self.grid_x,
            self.grid_y,
        ).permute(0, 3, 4, 2, 1)
        state = state.reshape(
            batch * self.grid_x * self.grid_y,
            hidden,
            rollout,
        )
        for block in self.temporal:
            state = block(state)
        state = state.view(
            batch,
            self.grid_x,
            self.grid_y,
            hidden,
            rollout,
        ).permute(0, 4, 3, 1, 2)
        state = state.reshape(
            batch * rollout,
            hidden,
            self.grid_x,
            self.grid_y,
        )
        state = self.head(state)
        state = state.view(
            batch,
            rollout,
            -1,
            self.grid_x,
            self.grid_y,
        ).permute(0, 2, 1, 3, 4)
        state = F.interpolate(
            state,
            size=(self.nt, self.nx, self.ny),
            mode="trilinear",
            align_corners=False,
        ).permute(0, 1, 3, 4, 2)

        skips = []
        scales = (
            skip_features,
            F.adaptive_avg_pool2d(
                skip_features,
                (max(1, self.nx // 2), max(1, self.ny // 2)),
            ),
            F.adaptive_avg_pool2d(
                skip_features,
                (max(1, self.nx // 4), max(1, self.ny // 4)),
            ),
        )
        for projection, features in zip(self.skip_projs, scales):
            mapped = projection(features)
            skips.append(
                F.interpolate(
                    mapped,
                    size=(self.nx, self.ny),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        gates = torch.sigmoid(
            self.skip_gate(context.mean(dim=1))
        ).view(batch, 3, 1, 1, 1, self.nt)
        return state + sum(
            skips[index].unsqueeze(-1) * gates[:, index]
            for index in range(3)
        )


class CoordAttentionOperator2D(nn.Module):
    def __init__(
        self,
        cfg: Operator2DConfig,
        nx: int,
        ny: int,
        nt: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.context = ContextEncoder2D(cfg)
        self.dynamics = LatentPseudoSpectralPDE2D(cfg)
        self.decoder = GridTrajectoryDecoder2D(cfg, nx, ny, nt)

    def encode_with_skips(self, u0: torch.Tensor):
        context, skip_features = self.context.forward_with_features(u0)
        return context, self.dynamics.rollout(u0), skip_features

    def forward_grid(self, u0: torch.Tensor) -> torch.Tensor:
        context, trajectory, skip_features = self.encode_with_skips(u0)
        return self.decoder(trajectory, context, skip_features)

    def reconstruct(self, u0: torch.Tensor) -> torch.Tensor:
        return self.dynamics.encoder.reconstruct(
            u0,
            (u0.size(-2), u0.size(-1)),
        )
