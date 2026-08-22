"""Versioned, resume-capable language-model checkpoints."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..config import ExperimentConfig, config_from_dict, config_to_dict
from ..tokenization import LanguageModelTokenizer

CHECKPOINT_SCHEMA_VERSION = 2
SUPPORTED_CHECKPOINT_SCHEMAS = {1, CHECKPOINT_SCHEMA_VERSION}


class CheckpointError(ValueError):
    """Raised when a checkpoint is unsupported or does not match the run."""


def capture_rng_state() -> dict[str, Any]:
    numpy_state = cast(tuple[str, np.ndarray[Any, Any], int, int, float], np.random.get_state())
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    numpy_payload: dict[str, Any] = state["numpy"]
    np.random.set_state(
        (
            numpy_payload["name"],
            numpy_payload["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_payload["position"]),
            int(numpy_payload["has_gauss"]),
            float(numpy_payload["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    cfg: ExperimentConfig,
    manifest_identity: str,
    tokenizer: LanguageModelTokenizer,
    trainer_state: dict[str, Any],
    data_order: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config_to_dict(cfg),
        "manifest_identity": manifest_identity,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "tokenizer_payload": tokenizer.to_payload(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "trainer_state": trainer_state,
        "data_order": data_order,
        "rng_state": capture_rng_state(),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise CheckpointError(f"unsupported checkpoint schema in {checkpoint_path}")
    required = {
        "config",
        "manifest_identity",
        "tokenizer_fingerprint",
        "tokenizer_payload",
        "model_state",
        "trainer_state",
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint is missing fields: {sorted(missing)}")
    return payload


def validate_resume_contract(
    payload: dict[str, Any],
    cfg: ExperimentConfig,
    data_order: dict[str, Any],
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("legacy checkpoints support evaluation and generation, but not exact training resume")
    checkpoint_cfg = config_from_dict(payload["config"])
    if checkpoint_cfg.model != cfg.model:
        raise CheckpointError("resume checkpoint model configuration does not match current configuration")
    if checkpoint_cfg.train != cfg.train:
        raise CheckpointError("resume checkpoint training configuration does not match current configuration")
    if payload.get("data_order") != data_order:
        raise CheckpointError("resume checkpoint data-order contract does not match the current loader")


def validate_checkpoint_identity(
    payload: dict[str, Any], manifest_identity: str, tokenizer: LanguageModelTokenizer
) -> None:
    if payload["manifest_identity"] != manifest_identity:
        raise CheckpointError("checkpoint manifest identity does not match prepared data")
    if payload["tokenizer_fingerprint"] != tokenizer.fingerprint():
        raise CheckpointError("checkpoint tokenizer fingerprint does not match prepared data")
