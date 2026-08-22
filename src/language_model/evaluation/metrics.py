"""Token-weighted loss and perplexity evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

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
    correct_top1 = 0
    correct_top5 = 0
    batch_count = 0
    pad_id = int(cast(Any, model).pad_id)
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
        valid = targets.ne(pad_id)
        valid_logits = output.logits[valid]
        valid_targets = targets[valid]
        correct_top1 += int(valid_logits.argmax(dim=-1).eq(valid_targets).sum().item())
        top_count = min(5, valid_logits.shape[-1])
        correct_top5 += int(
            valid_logits.topk(top_count, dim=-1).indices.eq(valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()
        )
        batch_count += 1
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("evaluation saw no non-padding target tokens")
    average_loss = total_loss / total_tokens
    return {
        "loss": average_loss,
        "negative_log_likelihood": total_loss,
        "perplexity": math.exp(average_loss) if average_loss < 80 else float("inf"),
        "accuracy_top1": correct_top1 / total_tokens,
        "accuracy_top5": correct_top5 / total_tokens,
        "tokens": total_tokens,
        "batches": batch_count,
    }


def enrich_text_normalized_metrics(metrics: dict[str, Any], split_metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    nll_bits = float(metrics["negative_log_likelihood"]) / math.log(2)
    characters = split_metadata.get("characters")
    utf8_bytes = split_metadata.get("utf8_bytes")
    if isinstance(characters, int) and characters > 0:
        result["bits_per_character"] = nll_bits / characters
        result["characters"] = characters
    if isinstance(utf8_bytes, int) and utf8_bytes > 0:
        result["bits_per_byte"] = nll_bits / utf8_bytes
        result["utf8_bytes"] = utf8_bytes
    return result
