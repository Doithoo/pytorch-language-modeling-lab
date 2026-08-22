"""End-to-end training orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, config_from_dict, config_to_dict, resolve_vocab_size
from ..data import TokenWindowDataset, default_manifest_path, load_manifest
from ..evaluation.metrics import evaluate_model
from ..generation import generate_text
from ..models import build_model
from ..preflight import environment_info, resolve_device
from ..tokenization import load_tokenizer
from ..utils import count_parameters, set_seed, sha256_file, worker_seed, write_json, write_yaml
from .checkpoint import load_checkpoint, validate_checkpoint_identity
from .trainer import Trainer


def _loader(
    dataset: TokenWindowDataset,
    cfg: ExperimentConfig,
    *,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(cfg.runtime.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.runtime.device in {"auto", "cuda"},
        worker_init_fn=worker_seed if cfg.data.num_workers else None,
        generator=generator,
        persistent_workers=cfg.data.num_workers > 0,
    )


def _validate_data_contract(cfg: ExperimentConfig, manifest: dict[str, Any]) -> None:
    if manifest["dataset_name"] != cfg.data.dataset_name:
        raise ValueError("config dataset_name does not match manifest")
    if manifest["context_length"] != cfg.data.context_length or manifest["stride"] != cfg.data.stride:
        raise ValueError("config window settings do not match manifest")
    if manifest["tokenizer"]["name"] != cfg.tokenizer.name:
        raise ValueError("config tokenizer does not match manifest")


def run_training(
    cfg: ExperimentConfig,
    *,
    manifest_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    path = Path(manifest_path) if manifest_path else default_manifest_path(cfg.data.manifest_dir, cfg.data.dataset_name)
    manifest = load_manifest(path)
    _validate_data_contract(cfg, manifest)
    tokenizer = load_tokenizer(path.parent / "tokenizer")
    if tokenizer.fingerprint() != manifest["tokenizer"]["fingerprint"]:
        raise ValueError("tokenizer fingerprint does not match manifest")
    cfg = resolve_vocab_size(cfg, tokenizer.vocab_size)
    device = resolve_device(cfg.runtime.device)
    set_seed(cfg.runtime.seed)

    train_dataset = TokenWindowDataset(path, "train", verify_hash=not dry_run)
    valid_dataset = TokenWindowDataset(path, "valid", verify_hash=not dry_run)
    train_loader = _loader(train_dataset, cfg, shuffle=True)
    valid_loader = _loader(valid_dataset, cfg, shuffle=False)
    model = build_model(cfg.model, tokenizer.pad_id).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    if dry_run:
        input_ids, targets = next(iter(train_loader))
        input_ids, targets = input_ids.to(device), targets.to(device)
        output = model(input_ids, targets)
        if output.loss is None:
            raise RuntimeError("dry run did not produce a loss")
        output.loss.backward()
        print(
            json.dumps(
                {
                    "phase": "dry_run",
                    "status": "completed",
                    "device": str(device),
                    "batch_shape": list(input_ids.shape),
                    "loss": float(output.loss.item()),
                    "tokens": output.token_count,
                    "parameters": total_parameters,
                }
            ),
            flush=True,
        )
        return None

    run_dir = cfg.runtime.output_dir / cfg.runtime.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and resume_path is None:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(run_dir / "tokenizer")
    write_yaml(run_dir / "config.yaml", config_to_dict(cfg))
    write_json(run_dir / "manifest.json", manifest)
    environment = environment_info(run_dir)
    write_json(run_dir / "environment.json", environment)
    write_json(
        run_dir / "run.json",
        {
            "status": "running",
            "device": str(device),
            "manifest_identity": manifest["identity"],
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=run_dir,
        manifest_identity=manifest["identity"],
        tokenizer=tokenizer,
    )
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location=device)
        validate_checkpoint_identity(payload, manifest["identity"], tokenizer)
        checkpoint_cfg = config_from_dict(payload["config"])
        if checkpoint_cfg.model != cfg.model:
            raise ValueError("resume checkpoint model configuration does not match current configuration")
        trainer.resume(payload)
    training_summary = trainer.fit()

    best_path = run_dir / "checkpoints" / "best.pt"
    best = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    test_dataset = TokenWindowDataset(path, "test", verify_hash=True)
    test_loader = _loader(test_dataset, cfg, shuffle=False)
    valid_metrics = evaluate_model(model, valid_loader, device, amp=cfg.train.amp)
    test_metrics = evaluate_model(model, test_loader, device, amp=cfg.train.amp)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    evaluation = {
        "manifest_identity": manifest["identity"],
        "checkpoint": str(best_path.name),
        "checkpoint_sha256": sha256_file(best_path),
        "valid": valid_metrics,
        "test": test_metrics,
    }
    write_json(evaluation_dir / "summary.json", evaluation)
    sample = generate_text(model, tokenizer, "The ", cfg.generation)
    (evaluation_dir / "prompts.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    final_summary = {
        **training_summary,
        "status": "completed",
        "device": str(device),
        "parameters": total_parameters,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "best_checkpoint_sha256": evaluation["checkpoint_sha256"],
        "valid": valid_metrics,
        "test": test_metrics,
    }
    write_json(run_dir / "run.json", final_summary)
    write_json(run_dir / "kaggle-run-summary.json", final_summary)
    return run_dir
