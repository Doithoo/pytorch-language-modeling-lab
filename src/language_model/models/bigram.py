"""Minimal bigram baseline for the next-token learning path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import ModelOutput, PastKeyValues


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, context_length: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.context_length = context_length
        self.transition = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        del past_key_values, use_cache
        logits = self.transition(input_ids)
        loss = None
        token_count = 0
        if targets is not None:
            token_count = int(targets.ne(self.pad_id).sum().item())
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=self.pad_id,
            )
        return ModelOutput(logits=logits, loss=loss, token_count=token_count)
