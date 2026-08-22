"""Built-in model construction and discovery."""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .bigram import BigramLanguageModel
from .transformer import TransformerLanguageModel

MODEL_NAMES = ("bigram", "transformer_decoder")


def build_model(cfg: ModelConfig, pad_id: int) -> nn.Module:
    if cfg.vocab_size is None:
        raise ValueError("model vocab_size must be resolved from the tokenizer")
    if cfg.name == "bigram":
        return BigramLanguageModel(cfg.vocab_size, pad_id, cfg.context_length)
    if cfg.name == "transformer_decoder":
        return TransformerLanguageModel(cfg, pad_id)
    raise ValueError(f"unknown model: {cfg.name}")


def list_models() -> list[dict[str, str]]:
    return [
        {"name": "bigram", "description": "token-to-next-token transition baseline"},
        {"name": "transformer_decoder", "description": "pre-LN causal decoder-only Transformer"},
    ]
