"""Sampling policies for autoregressive generation."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scores = logits / temperature
    if top_k is not None:
        keep = min(top_k, scores.shape[-1])
        threshold = torch.topk(scores, keep, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(1, sorted_indices, sorted_scores)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
