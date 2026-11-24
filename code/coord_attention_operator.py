"""多尺度潜空间 PDE 算子：u0(x) -> u(x, t)。

  KV：初值 u0 的上下文序列。卷积干路提取局部特征，按 token_count 个空间位置池化，
      加上可学习位置编码后经 Transformer 编码器做 token 间的自注意力。
  Q ：只来自物理坐标。x 用周期 Fourier 特征，t 用 Fourier 特征，
      分别投影后相加得到时空坐标嵌入；不拼接任何来自 u0 的信息。
  解码：支持坐标查询解码和低秩网格解码，可选周期常速平移物理分支。

默认坐标为物理单位：x ∈ [0, 1) 周期，t ∈ [0, 2]（见 burgers_common）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from burgers_common import NT, NX, T_FINAL, X_LENGTH


@dataclass
class OperatorConfig:
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 2
    token_count: int = 32
    conv_channels: int = 32
    x_bands: int = 16
    t_bands: int = 8
    dropout: float = 0.1
    dynamics: str = "none"
    latent_channels: int = 16
    latent_length: int = 64
    latent_recon: bool = False
    rollout_steps: int = 9
    pde_substeps: int = 2
    pde_modes: int = 24
    pde_layers: int = 0
    pde_rank: int = 8
    output_mode: str = "query"
    grid_hidden: int = 64
    grid_blocks: int = 2
    grid_length: int = 64
    grid_block: str = "dense"
    grid_last_depthwise: bool = False
    grid_last_rank: int = 0
    grid_lowrank_rank: int = 0
    grid_transport_velocity: float = 0.0
    grid_transport_blend: float = 0.0
    grid_transport_learnable: bool = False
    grid_transport_refine: bool = False
    grid_transport_refine_hidden: int = 0


class CoordinateEmbedding(nn.Module):
    """(x, t) -> d_model。x 用整数频率保证周期性；t 归一化到 [0, 1] 后用 2^k 频率。"""

    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        self.register_buffer("x_freqs", 2.0 * math.pi * torch.arange(1, cfg.x_bands + 1).float() / X_LENGTH)
        self.register_buffer("t_freqs", math.pi * 2.0 ** torch.arange(cfg.t_bands).float() / T_FINAL)
        self.x_proj = nn.Linear(2 * cfg.x_bands, cfg.d_model)
        self.t_proj = nn.Linear(2 * cfg.t_bands + 1, cfg.d_model)
        self.mix = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xf = x * self.x_freqs
        tf = t * self.t_freqs
        ex = self.x_proj(torch.cat([xf.sin(), xf.cos()], dim=-1))
        et = self.t_proj(torch.cat([tf.sin(), tf.cos(), t / T_FINAL], dim=-1))
        return self.mix(ex + et)


class ContextEncoder(nn.Module):
    """u0 [B, nx] -> 上下文 token [B, token_count, d_model]。"""

    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        c = cfg.conv_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, 5, padding=2, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(c, c, 5, padding=2, padding_mode="circular"), nn.SiLU(),
        )
        self.token_count = cfg.token_count
        self.proj = nn.Linear(c, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.token_count, cfg.d_model))
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, 2 * cfg.d_model, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.encoder_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward_with_features(self, u0: torch.Tensor):
        local = self.stem(u0.unsqueeze(1))
        features = F.adaptive_avg_pool1d(local, self.token_count)
        tokens = self.proj(features.transpose(1, 2)) + self.pos
        return self.norm(self.encoder(tokens)), local

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(u0)[0]


class CrossAttentionBlock(nn.Module):
    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(nn.Linear(cfg.d_model, 2 * cfg.d_model), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(2 * cfg.d_model, cfg.d_model))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return q + self.ffn(self.norm_f(q))


class LatentPDEEncoder(nn.Module):
    """u0 -> 周期潜空间网格 [B, C, L]。"""

    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        c = cfg.latent_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(c, c, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
        )
        self.decode = (
            nn.Sequential(
                nn.Conv1d(c, c, 3, padding=1, padding_mode="circular"),
                nn.SiLU(),
                nn.Conv1d(c, 1, 3, padding=1, padding_mode="circular"),
            )
            if cfg.latent_recon
            else None
        )
        self.latent_length = cfg.latent_length
        self.nx = NX

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        features = self.stem(u0.unsqueeze(1))
        return F.adaptive_avg_pool1d(features, self.latent_length)

    def reconstruct(self, u0: torch.Tensor) -> torch.Tensor:
        if self.decode is None:
            raise ValueError("latent reconstruction head is disabled")
        latent = self.forward(u0)
        restored = self.decode(latent)
        restored = F.interpolate(
            restored,
            size=self.nx,
            mode="linear",
            align_corners=False,
        )
        return restored.squeeze(1)


class LatentSpectralMixer(nn.Module):
    """Low-rank complex mixing across latent channels and Fourier modes."""

    def __init__(self, channels: int, modes: int, rank: int, length: int) -> None:
        super().__init__()
        self.length = length
        self.modes = modes
        self.rank = max(1, min(rank, channels))
        self.to_modes = nn.Conv1d(channels, self.rank, 1)
        self.out_mix = nn.Conv1d(channels, channels, 1)
        self.weight_real = nn.Parameter(
            torch.randn(self.rank, channels, modes) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(self.rank, channels, modes) * 0.02
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(self.to_modes(state), dim=-1)
        spectrum = spectrum[..., : self.modes]
        weight = torch.complex(self.weight_real, self.weight_imag)
        mixed = torch.einsum("brk,rck->bck", spectrum, weight)
        full = torch.zeros(
            mixed.shape[0],
            mixed.shape[1],
            self.length // 2 + 1,
            dtype=mixed.dtype,
            device=mixed.device,
        )
        full[..., : self.modes] = mixed
        return self.out_mix(torch.fft.irfft(full, n=self.length, dim=-1))


class LatentPDEBlock(nn.Module):
    """Residual local plus spectral update used inside the latent PDE RHS."""

    def __init__(self, channels: int, modes: int, rank: int, length: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.local = nn.Sequential(
            nn.Conv1d(
                channels, channels, 3, padding=1, padding_mode="circular"
            ),
            nn.SiLU(),
            nn.Conv1d(
                channels, channels, 3, padding=1, padding_mode="circular"
            ),
        )
        self.spectral = LatentSpectralMixer(channels, modes, rank, length)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        update = F.silu(self.local(self.norm(state)) + self.spectral(state))
        return self.proj(update)


class GridResBlock(nn.Module):
    def __init__(self, channels: int, kind: str = "dense") -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        if kind == "depthwise":
            self.body = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=1,
                    groups=channels,
                ),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 1),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.body(self.norm(state))


class LowRankGridResBlock(nn.Module):
    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        self.rank = max(1, min(rank, channels))
        self.norm = nn.GroupNorm(1, channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, self.rank, 3, padding=1, bias=False),
            nn.Conv2d(self.rank, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, self.rank, 3, padding=1, bias=False),
            nn.Conv2d(self.rank, channels, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.body(self.norm(state))


class GridTrajectoryDecoder(nn.Module):
    """Context-aware full-grid decoder for a latent trajectory."""

    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        c = cfg.latent_channels
        h = cfg.grid_hidden
        self.length = cfg.grid_length
        self.context_proj = nn.Linear(cfg.d_model, c)
        self.x_proj = nn.Linear(2 * cfg.x_bands, h)
        self.t_proj = nn.Linear(2 * cfg.t_bands + 1, h)
        self.input_proj = nn.Conv2d(c + h, h, 1)
        blocks = []
        block_count = max(1, cfg.grid_blocks)
        for index in range(block_count):
            kind = cfg.grid_block
            if cfg.grid_lowrank_rank > 0:
                blocks.append(LowRankGridResBlock(h, cfg.grid_lowrank_rank))
                continue
            if cfg.grid_last_rank > 0 and index == block_count - 1:
                blocks.append(LowRankGridResBlock(h, cfg.grid_last_rank))
                continue
            if cfg.grid_last_depthwise and index == block_count - 1:
                kind = "depthwise"
            blocks.append(GridResBlock(h, kind))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(h, 1, 1)
        self.skip_projs = nn.ModuleList(
            nn.Conv1d(cfg.conv_channels, 1, 1) for _ in range(3)
        )
        self.skip_gate = nn.Sequential(
            nn.Linear(cfg.d_model, h),
            nn.GELU(),
            nn.Linear(h, 3 * NT),
        )
        x = torch.arange(cfg.grid_length).float() / cfg.grid_length
        x_freq = 2.0 * math.pi * torch.arange(1, cfg.x_bands + 1).float()
        t = torch.linspace(0.0, 1.0, NT)
        t_freq = math.pi * 2.0 ** torch.arange(cfg.t_bands).float()
        x_phase = x[:, None] * x_freq[None, :]
        t_phase = t[:, None] * t_freq[None, :]
        self.register_buffer("x_features", torch.cat([x_phase.sin(), x_phase.cos()], dim=-1))
        self.register_buffer("t_features", torch.cat([t_phase.sin(), t_phase.cos(), t[:, None]], dim=-1))

    def forward(
        self,
        trajectory: torch.Tensor,
        context: torch.Tensor,
        skip_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_tokens = context
        context = self.context_proj(context).transpose(1, 2)
        context = F.interpolate(
            context,
            size=trajectory.size(2),
            mode="linear",
            align_corners=False,
        )
        state = trajectory + context.unsqueeze(-1)
        state = F.interpolate(
            state,
            size=(self.length, NT),
            mode="bilinear",
            align_corners=False,
        )
        x_pos = self.x_proj(self.x_features).transpose(0, 1)[None, :, :, None]
        t_pos = self.t_proj(self.t_features).transpose(0, 1)[None, :, None, :]
        position = x_pos + t_pos
        state = self.input_proj(
            torch.cat([state, position.expand(state.size(0), -1, -1, -1)], dim=1)
        )
        for block in self.blocks:
            state = block(state)
        field = self.head(state)
        field = F.interpolate(
            field,
            size=(NX, NT),
            mode="bilinear",
            align_corners=False,
        )
        field = field.squeeze(1)
        if skip_features is None:
            return field
        scales = [
            skip_features,
            F.adaptive_avg_pool1d(skip_features, NX // 2),
            F.adaptive_avg_pool1d(skip_features, NX // 4),
        ]
        skip_maps = []
        for projection, features in zip(self.skip_projs, scales):
            skip_map = projection(features)
            skip_maps.append(
                F.interpolate(skip_map, size=NX, mode="linear", align_corners=False)
            )
        gates = torch.sigmoid(
            self.skip_gate(context_tokens.mean(dim=1))
        ).view(context.size(0), 3, NT)
        skips = sum(
            skip_maps[index].unsqueeze(-1)
            * gates[:, index].view(context.size(0), 1, 1, NT)
            for index in range(3)
        )
        return field + skips.squeeze(1)


class LatentPseudoSpectralPDE(nn.Module):
    """Burgers 风格的潜空间神经 PDE。

    线性平流和扩散用 Fourier 指数积分精确推进，非线性反应项显式推进。
    """

    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        c = cfg.latent_channels
        length = cfg.latent_length
        self.length = length
        self.rollout_steps = max(2, cfg.rollout_steps)
        self.substeps = max(1, cfg.pde_substeps)
        self.modes = min(max(1, cfg.pde_modes), length // 2 + 1)
        self.encoder = LatentPDEEncoder(cfg)
        self.nonlinear = nn.Conv1d(
            c, c, 3, padding=1, padding_mode="circular"
        )
        self.reaction = nn.Conv1d(c, c, 1)
        nn.init.zeros_(self.reaction.weight)
        nn.init.zeros_(self.reaction.bias)
        self.blocks = nn.ModuleList(
            LatentPDEBlock(c, self.modes, cfg.pde_rank, length)
            for _ in range(max(0, cfg.pde_layers))
        )
        self.log_advection = nn.Parameter(torch.zeros(c))
        self.log_diffusion = nn.Parameter(
            torch.full((c,), math.log(math.expm1(0.01)))
        )
        frequencies = 2.0 * math.pi * torch.fft.rfftfreq(length)
        self.register_buffer("frequencies", frequencies)
        mode_mask = torch.zeros(length // 2 + 1)
        mode_mask[: self.modes] = 1.0
        self.register_buffer("mask", mode_mask)

    def _linear_step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        advection = 0.5 * torch.tanh(self.log_advection)
        diffusion = F.softplus(self.log_diffusion)[None, :, None]
        linear = (
            -1j * self.frequencies[None, None, :] * advection[None, :, None]
            - diffusion * self.frequencies[None, None, :].square()
        )
        spectrum = torch.fft.rfft(state, dim=-1) * self.mask
        return torch.fft.irfft(
            spectrum * torch.exp(linear * dt),
            n=self.length,
            dim=-1,
        )

    def _nonlinear(self, state: torch.Tensor) -> torch.Tensor:
        update = 0.1 * torch.tanh(self.nonlinear(state)) + self.reaction(state)
        for block in self.blocks:
            update = update + block(state)
        return update

    def _step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        linear = self._linear_step(state, dt)
        nonlinear = self._nonlinear(state)
        state = linear + dt * nonlinear
        spectrum = torch.fft.rfft(state, dim=-1) * self.mask
        return torch.fft.irfft(spectrum, n=self.length, dim=-1)

    def rollout(self, u0: torch.Tensor) -> torch.Tensor:
        """返回 [B, C, L, K] 的整条潜空间轨迹。"""
        state = self.encoder(u0)
        states = [state]
        dt = T_FINAL / (self.rollout_steps - 1)
        step_dt = dt / self.substeps
        for _ in range(1, self.rollout_steps):
            for _ in range(self.substeps):
                state = self._step(state, step_dt)
            states.append(state)
        return torch.stack(states, dim=-1)

    def reconstruct(self, u0: torch.Tensor) -> torch.Tensor:
        return self.encoder.reconstruct(u0)

    def sample(
        self,
        trajectory: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """在 (x, t) 处双线性读取潜空间轨迹，返回 [B, P, C]。"""
        batch, channels, length, time_steps = trajectory.shape
        x = x.squeeze(-1).remainder(X_LENGTH) / X_LENGTH
        t = t.squeeze(-1).clamp(0.0, T_FINAL) / T_FINAL
        x_position = x * length
        t_position = t * (time_steps - 1)

        x0 = torch.floor(x_position).long().remainder(length)
        x1 = (x0 + 1).remainder(length)
        x_frac = x_position - torch.floor(x_position)
        t0 = torch.floor(t_position).long().clamp(0, time_steps - 2)
        t1 = t0 + 1
        t_frac = t_position - t0.float()
        flat = trajectory.reshape(batch, channels, length * time_steps)

        def gather(x_index: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
            index = x_index * time_steps + t_index
            index = index[:, None, :].expand(-1, channels, -1)
            return torch.gather(flat, 2, index)

        value00 = gather(x0, t0)
        value01 = gather(x0, t1)
        value10 = gather(x1, t0)
        value11 = gather(x1, t1)
        lower = value00 + x_frac[:, None, :] * (value10 - value00)
        upper = value01 + x_frac[:, None, :] * (value11 - value01)
        sampled = lower + t_frac[:, None, :] * (upper - lower)
        return sampled.transpose(1, 2)


class CoordAttentionOperator(nn.Module):
    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context = ContextEncoder(cfg)
        grid_mode = cfg.output_mode == "grid"
        self.coords = None if grid_mode else CoordinateEmbedding(cfg)
        self.dynamics = (
            LatentPseudoSpectralPDE(cfg)
            if cfg.dynamics == "latent_pde"
            else None
        )
        self.latent_query = (
            nn.Linear(cfg.latent_channels, cfg.d_model)
            if self.dynamics is not None and not grid_mode
            else None
        )
        self.grid_decoder = (
            GridTrajectoryDecoder(cfg)
            if self.dynamics is not None and cfg.output_mode == "grid"
            else None
        )
        self.blocks = (
            None
            if grid_mode
            else nn.ModuleList(
                CrossAttentionBlock(cfg) for _ in range(cfg.decoder_layers)
            )
        )
        self.head = (
            None
            if grid_mode
            else nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )
        )
        if cfg.grid_transport_velocity != 0.0:
            x = torch.arange(NX).float()
            t = torch.linspace(0.0, T_FINAL, NT)
            position = (
                x[:, None]
                - cfg.grid_transport_velocity
                * t[None, :]
                * NX
                / X_LENGTH
            ).remainder(NX)
            x0 = torch.floor(position).long()
            self.register_buffer("transport_x0", x0)
            self.register_buffer("transport_x1", (x0 + 1).remainder(NX))
            self.register_buffer("transport_frac", position - x0.float())
        self.transport_blend = (
            nn.Parameter(torch.tensor(float(cfg.grid_transport_blend)))
            if cfg.grid_transport_learnable
            else None
        )
        if cfg.grid_transport_refine_hidden > 0:
            self.transport_refiner = nn.Sequential(
                nn.Conv2d(1, cfg.grid_transport_refine_hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(cfg.grid_transport_refine_hidden, 1, 3, padding=1),
            )
        elif cfg.grid_transport_refine:
            self.transport_refiner = nn.Conv2d(1, 1, 3, padding=1)
        else:
            self.transport_refiner = None
        if self.transport_refiner is not None:
            final = (
                self.transport_refiner[-1]
                if isinstance(self.transport_refiner, nn.Sequential)
                else self.transport_refiner
            )
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def encode(self, u0: torch.Tensor):
        return self.context(u0), (
            self.dynamics.rollout(u0) if self.dynamics is not None else None
        )

    def encode_with_skips(self, u0: torch.Tensor):
        context, skip_features = self.context.forward_with_features(u0)
        return context, (
            self.dynamics.rollout(u0) if self.dynamics is not None else None
        ), skip_features

    def decode(
        self,
        kv: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        trajectory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """kv [B, T, d]；x, t [B, P, 1] -> u [B, P]。"""
        if self.coords is None or self.blocks is None or self.head is None:
            raise ValueError("query decoder is disabled in grid output mode")
        q = self.coords(x, t)
        if self.dynamics is not None:
            if trajectory is None:
                raise ValueError("latent_pde requires a latent trajectory")
            q = q + self.latent_query(self.dynamics.sample(trajectory, x, t))
        for block in self.blocks:
            q = block(q, kv)
        return self.head(q).squeeze(-1)

    def forward(self, u0: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.grid_decoder is not None:
            raise ValueError("use forward_grid for full-grid output mode")
        kv, trajectory = self.encode(u0)
        return self.decode(kv, x, t, trajectory)

    def forward_grid(self, u0: torch.Tensor) -> torch.Tensor:
        if self.grid_decoder is None:
            raise ValueError("grid decoder is not enabled")
        context, trajectory, skip_features = self.encode_with_skips(u0)
        field = self.grid_decoder(trajectory, context, skip_features)
        velocity = self.cfg.grid_transport_velocity
        if velocity == 0.0:
            return field
        source = u0.unsqueeze(-1).expand(-1, -1, NT)
        index0 = self.transport_x0.unsqueeze(0).expand(u0.size(0), -1, -1)
        index1 = self.transport_x1.unsqueeze(0).expand(u0.size(0), -1, -1)
        lower = torch.gather(source, 1, index0)
        upper = torch.gather(source, 1, index1)
        transport = lower + self.transport_frac * (upper - lower)
        blend = (
            self.transport_blend
            if self.transport_blend is not None
            else self.cfg.grid_transport_blend
        )
        residual = field - transport
        if self.transport_refiner is not None:
            residual = residual + self.transport_refiner(
                residual.unsqueeze(1)
            ).squeeze(1)
        return transport + blend * residual

    def reconstruct(self, u0: torch.Tensor) -> torch.Tensor:
        if self.dynamics is None:
            raise ValueError("reconstruction requires latent dynamics")
        return self.dynamics.reconstruct(u0)
