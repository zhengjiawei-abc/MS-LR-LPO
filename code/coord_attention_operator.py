"""坐标查询交叉注意力算子：u0(x) -> u(x, t)。

  KV：初值 u0 的上下文序列。卷积干路提取局部特征，按 token_count 个空间位置池化，
      加上可学习位置编码后经 Transformer 编码器做 token 间的自注意力。
  Q ：只来自物理坐标。x 用周期 Fourier 特征，t 用 Fourier 特征，
      分别投影后相加得到时空坐标嵌入；不拼接任何来自 u0 的信息。
  解码：若干层 [坐标 Q 对上下文 KV 的交叉注意力 + FFN]，再经 MLP 头输出 u(x, t)。

坐标为物理单位：x ∈ [0, 1) 周期，t ∈ [0, 2]（见 burgers_common）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from burgers_common import T_FINAL, X_LENGTH


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

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        features = F.adaptive_avg_pool1d(self.stem(u0.unsqueeze(1)), self.token_count)
        tokens = self.proj(features.transpose(1, 2)) + self.pos
        return self.norm(self.encoder(tokens))


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


class CoordAttentionOperator(nn.Module):
    def __init__(self, cfg: OperatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context = ContextEncoder(cfg)
        self.coords = CoordinateEmbedding(cfg)
        self.blocks = nn.ModuleList(CrossAttentionBlock(cfg) for _ in range(cfg.decoder_layers))
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model),
                                  nn.GELU(), nn.Linear(cfg.d_model, 1))

    def decode(self, kv: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """kv [B, T, d]；x, t [B, P, 1] -> u [B, P]。"""
        q = self.coords(x, t)
        for block in self.blocks:
            q = block(q, kv)
        return self.head(q).squeeze(-1)

    def forward(self, u0: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.decode(self.context(u0), x, t)
