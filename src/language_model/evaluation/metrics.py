"""Token-weighted loss and perplexity evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int | None = None,
    amp: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    for input_ids, targets in batches:
        if max_batches is not None and batch_count >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(input_ids, targets)
        if output.loss is None:
            raise RuntimeError("model did not return loss during evaluation")
        total_loss += float(output.loss.item()) * output.token_count
        total_tokens += output.token_count
        batch_count += 1
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("evaluation saw no non-padding target tokens")
    average_loss = total_loss / total_tokens
    return {
        "loss": average_loss,
        "perplexity": math.exp(average_loss) if average_loss < 80 else float("inf"),
        "tokens": total_tokens,
        "batches": batch_count,
    }
