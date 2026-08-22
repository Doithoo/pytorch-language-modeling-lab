"""Evaluate a saved checkpoint against a verified prepared split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ..config import config_from_dict
from ..data import TokenWindowDataset, load_manifest
from ..models import build_model
from ..preflight import resolve_device
from ..tokenization import tokenizer_from_payload
from ..training.checkpoint import load_checkpoint, validate_checkpoint_identity
from .metrics import enrich_text_normalized_metrics, evaluate_model


def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
    device: str = "auto",
    max_batches: int | None = None,
    processed_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = resolve_device(device)
    payload = load_checkpoint(checkpoint, map_location=resolved_device)
    cfg = config_from_dict(payload["config"])
    manifest = load_manifest(manifest_path)
    tokenizer = tokenizer_from_payload(payload["tokenizer_payload"])
    validate_checkpoint_identity(payload, manifest["identity"], tokenizer)
    dataset = TokenWindowDataset(manifest_path, split, processed_root=processed_root, verify_hash=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    model = build_model(cfg.model, tokenizer.pad_id).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    metrics = evaluate_model(
        model,
        loader,
        resolved_device,
        max_batches=max_batches,
        amp=cfg.train.amp,
    )
    if max_batches is None:
        metrics = enrich_text_normalized_metrics(metrics, manifest["splits"][split])
    return {
        "checkpoint": str(Path(checkpoint)),
        "manifest_identity": manifest["identity"],
        "split": split,
        "device": str(resolved_device),
        **metrics,
    }
