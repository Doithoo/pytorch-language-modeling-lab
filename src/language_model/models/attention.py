"""Readable causal multi-head self-attention with optional KV caching."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import PastKeyValue


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        implementation: str = "readable",
        rotary: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if implementation not in {"readable", "sdpa"}:
            raise ValueError(f"unknown attention implementation: {implementation}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.implementation = implementation
        self.rotary = rotary
        if rotary and self.head_dim % 2:
            raise ValueError("rotary positional encoding requires an even attention head dimension")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def _apply_rotary(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self.rotary:
            return tensor
        half = self.head_dim // 2
        frequencies = torch.arange(half, device=tensor.device, dtype=torch.float32)
        inverse = 1.0 / (10000.0 ** (frequencies / half))
        angles = positions.to(torch.float32).unsqueeze(-1) * inverse.unsqueeze(0)
        cos = angles.cos().to(dtype=tensor.dtype).view(1, 1, -1, half)
        sin = angles.sin().to(dtype=tensor.dtype).view(1, 1, -1, half)
        even = tensor[..., ::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape[:2] != (batch, self.n_heads) or past_key.shape != past_value.shape:
                raise ValueError("invalid past_key_value shape")
            past_length = past_key.shape[2]
        current_positions = torch.arange(past_length, past_length + time, device=x.device)
        query = self._apply_rotary(query, current_positions)
        key = self._apply_rotary(key, current_positions)

        if past_key_value is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        key_positions = torch.arange(key.shape[2], device=x.device)
        query_positions = torch.arange(past_length, past_length + time, device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if self.implementation == "sdpa":
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask.view(1, 1, time, key.shape[2]),
                dropout_p=self.attention_dropout.p if self.training else 0.0,
            )
        else:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~causal_mask.view(1, 1, time, key.shape[2]), torch.finfo(scores.dtype).min)
            attention = F.softmax(scores, dim=-1)
            attention = self.attention_dropout(attention)
            output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch, time, channels)
        output = self.residual_dropout(self.output(output))
        present = (key, value) if use_cache else None
        return output, present
