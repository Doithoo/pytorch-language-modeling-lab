"""Checkpoint-backed model loading and text generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..config import GenerationConfig, config_from_dict
from ..generation import generate_text
from ..models import build_model
from ..preflight import resolve_device
from ..tokenization import LanguageModelTokenizer, tokenizer_from_payload
from ..training.checkpoint import CheckpointError, load_checkpoint


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: LanguageModelTokenizer,
        generation_config: GenerationConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, device: str = "auto") -> Predictor:
        resolved_device = resolve_device(device)
        payload = load_checkpoint(checkpoint, map_location=resolved_device)
        cfg = config_from_dict(payload["config"])
        tokenizer = tokenizer_from_payload(payload["tokenizer_payload"])
        if tokenizer.fingerprint() != payload["tokenizer_fingerprint"]:
            raise CheckpointError("embedded tokenizer does not match checkpoint fingerprint")
        model = build_model(cfg.model, tokenizer.pad_id)
        model.load_state_dict(payload["model_state"])
        return cls(model, tokenizer, cfg.generation, resolved_device)

    def generate(self, prompt: str, **overrides: Any) -> dict[str, object]:
        config = self.generation_config
        allowed = {"max_new_tokens", "temperature", "top_k", "top_p", "use_cache", "seed"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unknown generation options: {sorted(unknown)}")
        config = replace(config, **overrides)
        return generate_text(self.model, self.tokenizer, prompt, config)
