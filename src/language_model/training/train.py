"""End-to-end training orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, config_to_dict, resolve_vocab_size
from ..data import TokenWindowDataset, default_manifest_path, load_manifest
from ..evaluation.metrics import enrich_text_normalized_metrics, evaluate_model
from ..generation import generate_text
from ..models import build_model
from ..preflight import environment_info, resolve_device
from ..tokenization import load_tokenizer
from ..utils import atomic_write_text, count_parameters, set_seed, sha256_file, worker_seed, write_json, write_yaml
from .checkpoint import load_checkpoint, validate_checkpoint_identity, validate_resume_contract
from .sampler import ResumableBatchSampler
from .trainer import Trainer


def _loader(
    dataset: TokenWindowDataset,
    cfg: ExperimentConfig,
    *,
    shuffle: bool,
    device: torch.device,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], ResumableBatchSampler]:
    sampler = ResumableBatchSampler(
        len(dataset),
        cfg.train.batch_size,
        seed=cfg.runtime.seed,
        shuffle=shuffle,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed if cfg.data.num_workers else None,
        generator=torch.Generator().manual_seed(cfg.runtime.seed),
        persistent_workers=cfg.data.num_workers > 0,
    )
    return loader, sampler


def _validate_data_contract(cfg: ExperimentConfig, manifest: dict[str, Any]) -> None:
    if manifest["dataset_name"] != cfg.data.dataset_name:
        raise ValueError("config dataset_name does not match manifest")
    if manifest["context_length"] != cfg.data.context_length or manifest["stride"] != cfg.data.stride:
        raise ValueError("config window settings do not match manifest")
    if manifest["tokenizer"]["name"] != cfg.tokenizer.name:
        raise ValueError("config tokenizer does not match manifest")
    if manifest.get("split_strategy", "document_count") != cfg.data.split_strategy:
        raise ValueError("config split_strategy does not match manifest")
    if manifest.get("group_by_source", False) != cfg.data.group_by_source:
        raise ValueError("config group_by_source does not match manifest")


def run_training(
    cfg: ExperimentConfig,
    *,
    manifest_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    dry_run: bool = False,
    stop_after_steps: int | None = None,
) -> Path | None:
    path = Path(manifest_path) if manifest_path else default_manifest_path(cfg.data.manifest_dir, cfg.data.dataset_name)
    manifest = load_manifest(path)
    _validate_data_contract(cfg, manifest)
    tokenizer = load_tokenizer(path.parent / "tokenizer")
    if tokenizer.fingerprint() != manifest["tokenizer"]["fingerprint"]:
        raise ValueError("tokenizer fingerprint does not match manifest")
    cfg = resolve_vocab_size(cfg, tokenizer.vocab_size)
    device = resolve_device(cfg.runtime.device)
    set_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)

    train_dataset = TokenWindowDataset(path, "train", verify_hash=not dry_run)
    valid_dataset = TokenWindowDataset(path, "valid", verify_hash=not dry_run)
    train_loader, train_sampler = _loader(train_dataset, cfg, shuffle=True, device=device)
    valid_loader, _ = _loader(valid_dataset, cfg, shuffle=False, device=device)
    model = build_model(cfg.model, tokenizer.pad_id).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = load_checkpoint(resume_path, map_location=device)
        validate_checkpoint_identity(resume_payload, manifest["identity"], tokenizer)
        validate_resume_contract(resume_payload, cfg, train_sampler.contract())

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
    if run_dir.exists() and any(run_dir.iterdir()) and resume_payload is None:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    if resume_payload is not None and float(resume_payload["trainer_state"]["best_val_loss"]) < float("inf"):
        expected_best = run_dir / "checkpoints" / "best.pt"
        if not expected_best.is_file():
            write_json(
                run_dir / "resume-warning.json",
                {
                    "warning": "historical best checkpoint was not present; final evaluation will use the resumed last checkpoint",
                    "resume_checkpoint": str(resume_path),
                },
            )
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
        train_sampler=train_sampler,
    )
    if resume_payload is not None:
        trainer.resume(resume_payload)
    try:
        training_summary = trainer.fit(stop_after_steps=stop_after_steps)
        if stop_after_steps is not None and training_summary["status"] == "interrupted":
            write_json(
                run_dir / "run.json",
                {
                    "status": "interrupted",
                    "device": str(device),
                    "manifest_identity": manifest["identity"],
                    "global_step": trainer.global_step,
                    "micro_step": trainer.micro_step,
                    "processed_tokens": trainer.processed_tokens,
                },
            )
            return run_dir

        best_path = run_dir / "checkpoints" / "best.pt"
        if not best_path.is_file():
            best_path = run_dir / "checkpoints" / "last.pt"
        best = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(best["model_state"])
        test_dataset = TokenWindowDataset(path, "test", verify_hash=True)
        test_loader, _ = _loader(test_dataset, cfg, shuffle=False, device=device)
        valid_metrics = enrich_text_normalized_metrics(
            evaluate_model(model, valid_loader, device, amp=cfg.train.amp),
            manifest["splits"]["valid"],
        )
        test_metrics = enrich_text_normalized_metrics(
            evaluate_model(model, test_loader, device, amp=cfg.train.amp),
            manifest["splits"]["test"],
        )
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
        samples = [generate_text(model, tokenizer, prompt, cfg.generation) for prompt in cfg.generation.prompts]
        atomic_write_text(
            evaluation_dir / "prompts.jsonl",
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        )
        final_summary = {
            **training_summary,
            "status": "completed",
            "device": str(device),
            "manifest_identity": manifest["identity"],
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "best_checkpoint": {
                "step": int(best["trainer_state"]["global_step"]),
                "sha256": evaluation["checkpoint_sha256"],
            },
            "valid": valid_metrics,
            "test": test_metrics,
        }
        write_json(run_dir / "run.json", final_summary)
        write_json(run_dir / "kaggle-run-summary.json", final_summary)
        return run_dir
    except BaseException as exc:
        write_json(
            run_dir / "run.json",
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "device": str(device),
                "manifest_identity": manifest["identity"],
                "global_step": trainer.global_step,
                "micro_step": trainer.micro_step,
                "processed_tokens": trainer.processed_tokens,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
