"""Small pre-LN decoder-only Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .attention import CausalSelfAttention
from .spec import ModelOutput, PastKeyValue, PastKeyValues


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.mlp_norm = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        attended, present = self.attention(
            self.attention_norm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attended
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class TransformerLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int) -> None:
        super().__init__()
        if cfg.vocab_size is None:
            raise ValueError("model vocab_size must be resolved before construction")
        self.config = cfg
        self.vocab_size = cfg.vocab_size
        self.context_length = cfg.context_length
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}")
        batch, time = input_ids.shape
        del batch
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per decoder block")
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        if past_length + time > self.context_length:
            raise ValueError(f"sequence length {past_length + time} exceeds context length {self.context_length}")
        positions = torch.arange(past_length, past_length + time, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        x = self.dropout(x)

        presents: list[PastKeyValue] = []
        for index, block in enumerate(self.blocks):
            past = past_key_values[index] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        token_count = 0
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same [batch, time] shape as input_ids")
            token_count = int(targets.ne(self.pad_id).sum().item())
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=self.pad_id,
            )
        return ModelOutput(
            logits=logits,
            loss=loss,
            token_count=token_count,
            past_key_values=tuple(presents) if use_cache else None,
        )
