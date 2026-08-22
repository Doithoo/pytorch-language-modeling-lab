"""Autoregressive generation with cache/no-cache parity."""

from __future__ import annotations

from typing import Any, cast

import torch

from ..config import GenerationConfig
from ..tokenization import LanguageModelTokenizer
from .sampling import sample_next_token


@torch.inference_mode()
def generate_ids(
    model: torch.nn.Module,
    prompt_ids: list[int],
    cfg: GenerationConfig,
    *,
    eos_id: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    device = next(model.parameters()).device
    context_length = int(cast(Any, model).context_length)
    generated = list(prompt_ids)
    model.eval()
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(cfg.seed)
    past = None
    cache_used = False
    for _ in range(cfg.max_new_tokens):
        if cfg.use_cache and past is not None:
            current = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        else:
            current = torch.tensor([generated[-context_length:]], dtype=torch.long, device=device)
        output = model(current, past_key_values=past, use_cache=cfg.use_cache)
        next_token = sample_next_token(
            output.logits[:, -1, :],
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            generator=generator,
        )
        token_id = int(next_token.item())
        generated.append(token_id)
        past = output.past_key_values if cfg.use_cache else None
        cache_used = cache_used or past is not None
        if past is not None and past[0][0].shape[2] >= context_length:
            past = None
        if eos_id is not None and token_id == eos_id:
            break
    return generated, {"new_tokens": len(generated) - len(prompt_ids), "cache_used": cache_used}


def generate_text(
    model: torch.nn.Module,
    tokenizer: LanguageModelTokenizer,
    prompt: str,
    cfg: GenerationConfig,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    ids, metadata = generate_ids(model, prompt_ids, cfg, eos_id=tokenizer.eos_id)
    return {
        "prompt": prompt,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "token_ids": ids,
        "generation": {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "seed": cfg.seed,
            "use_cache": cfg.use_cache,
            **metadata,
        },
    }
