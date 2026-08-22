"""AdamW and warmup-cosine learning-rate construction."""

from __future__ import annotations

import math

import torch

from ..config import TrainConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return max(1e-8, (step + 1) / cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
