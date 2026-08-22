"""Shared model output and KV-cache contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[PastKeyValue, ...]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    token_count: int = 0
    past_key_values: PastKeyValues | None = None
