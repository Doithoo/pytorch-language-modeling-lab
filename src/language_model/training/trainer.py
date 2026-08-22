"""Step-based trainer with AMP, accumulation, validation, and resume."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..evaluation.metrics import evaluate_model
from ..tokenization import LanguageModelTokenizer
from .checkpoint import restore_rng_state, save_checkpoint
from .optimizer import build_optimizer, build_scheduler
from .sampler import ResumableBatchSampler


def _build_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        cfg: ExperimentConfig,
        device: torch.device,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        valid_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        output_dir: str | Path,
        manifest_identity: str,
        tokenizer: LanguageModelTokenizer,
        train_sampler: ResumableBatchSampler,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_identity = manifest_identity
        self.tokenizer = tokenizer
        self.train_sampler = train_sampler
        self.optimizer = build_optimizer(model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train)
        self.use_amp = cfg.train.amp and device.type == "cuda"
        self.scaler = _build_scaler(self.use_amp)
        self.global_step = 0
        self.micro_step = 0
        self.best_val_loss = float("inf")
        self.processed_tokens = 0
        self.started_at = time.monotonic()
        self.metrics_path = self.output_dir / "metrics.csv"
        self._ensure_metrics_file()

    def _ensure_metrics_file(self) -> None:
        if self.metrics_path.exists():
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "train_loss",
                    "valid_loss",
                    "valid_perplexity",
                    "valid_accuracy_top1",
                    "valid_accuracy_top5",
                    "learning_rate",
                    "processed_tokens",
                    "elapsed_seconds",
                    "tokens_per_second",
                    "peak_memory_bytes",
                ],
            )
            writer.writeheader()

    def _state(self) -> dict[str, Any]:
        epoch, batch_in_epoch = divmod(self.micro_step, self.train_sampler.batches_per_epoch)
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "best_val_loss": self.best_val_loss,
            "processed_tokens": self.processed_tokens,
            "elapsed_seconds": time.monotonic() - self.started_at,
        }

    def resume(self, payload: dict[str, Any]) -> None:
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.scaler.load_state_dict(payload.get("scaler_state", {}))
        state = payload["trainer_state"]
        self.global_step = int(state["global_step"])
        self.micro_step = int(state.get("micro_step", self.global_step * self.cfg.train.gradient_accumulation_steps))
        self.train_sampler.set_start_micro_step(self.micro_step)
        self.best_val_loss = float(state["best_val_loss"])
        self.processed_tokens = int(state["processed_tokens"])
        self.started_at = time.monotonic() - float(state.get("elapsed_seconds", 0.0))
        restore_rng_state(payload["rng_state"])

    def _save(self, name: str) -> None:
        save_checkpoint(
            self.checkpoint_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            cfg=self.cfg,
            manifest_identity=self.manifest_identity,
            tokenizer=self.tokenizer,
            trainer_state=self._state(),
            data_order=self.train_sampler.contract(),
        )

    def _append_metrics(self, train_loss: float, validation: dict[str, Any]) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        peak_memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        row: dict[str, Any] = {
            "step": self.global_step,
            "train_loss": train_loss,
            "valid_loss": validation["loss"],
            "valid_perplexity": validation["perplexity"],
            "valid_accuracy_top1": validation["accuracy_top1"],
            "valid_accuracy_top5": validation["accuracy_top5"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "processed_tokens": self.processed_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self.processed_tokens / max(elapsed, 1e-9),
            "peak_memory_bytes": peak_memory,
        }
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writerow(row)
        print(json.dumps({"phase": "training", "status": "running", **row}), flush=True)
        return row

    def fit(self, *, stop_after_steps: int | None = None) -> dict[str, Any]:
        self.model.train()
        self.train_sampler.set_start_micro_step(self.micro_step)
        iterator = iter(self.train_loader)
        interval_loss = 0.0
        interval_tokens = 0
        last_metrics: dict[str, Any] = {}
        target_step = self.cfg.train.max_steps
        if stop_after_steps is not None:
            if stop_after_steps < self.global_step:
                raise ValueError("stop_after_steps cannot be below the current global step")
            target_step = min(target_step, stop_after_steps)
        while self.global_step < target_step:
            self.optimizer.zero_grad(set_to_none=True)
            micro_batches: list[tuple[torch.Tensor, torch.Tensor, int]] = []
            for _ in range(self.cfg.train.gradient_accumulation_steps):
                try:
                    input_ids, targets = next(iterator)
                except StopIteration:
                    iterator = iter(self.train_loader)
                    input_ids, targets = next(iterator)
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                token_count = int(targets.ne(self.tokenizer.pad_id).sum().item())
                micro_batches.append((input_ids, targets, token_count))
                self.micro_step += 1
            accumulated_tokens = sum(item[2] for item in micro_batches)
            if accumulated_tokens == 0:
                raise RuntimeError("gradient accumulation saw no non-padding target tokens")

            for input_ids, targets, expected_tokens in micro_batches:
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = self.model(input_ids, targets)
                    if output.loss is None:
                        raise RuntimeError("model did not return a training loss")
                    if output.token_count != expected_tokens:
                        raise RuntimeError("model token count does not match the training batch")
                    scaled_loss = output.loss * (output.token_count / accumulated_tokens)
                self.scaler.scale(scaled_loss).backward()
                interval_loss += float(output.loss.item()) * output.token_count
                interval_tokens += output.token_count
                self.processed_tokens += output.token_count

            self.scaler.unscale_(self.optimizer)
            if self.cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            should_evaluate = (
                self.global_step % self.cfg.train.eval_interval == 0 or self.global_step == self.cfg.train.max_steps
            )
            should_save = self.global_step % self.cfg.train.save_interval == 0
            if should_evaluate:
                validation = evaluate_model(
                    self.model,
                    self.valid_loader,
                    self.device,
                    max_batches=self.cfg.train.eval_batches,
                    amp=self.use_amp,
                )
                train_loss = interval_loss / max(1, interval_tokens)
                last_metrics = self._append_metrics(train_loss, validation)
                interval_loss = 0.0
                interval_tokens = 0
                if validation["loss"] < self.best_val_loss:
                    self.best_val_loss = float(validation["loss"])
                    self._save("best.pt")
                self._save("last.pt")
            elif should_save:
                self._save("last.pt")
            elif self.global_step % self.cfg.train.log_interval == 0:
                elapsed = time.monotonic() - self.started_at
                print(
                    json.dumps(
                        {
                            "phase": "training",
                            "status": "heartbeat",
                            "step": self.global_step,
                            "processed_tokens": self.processed_tokens,
                            "elapsed_seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
        interrupted = self.global_step < self.cfg.train.max_steps
        if interrupted:
            self._save("last.pt")
        return {
            "status": "interrupted" if interrupted else "completed",
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "best_val_loss": self.best_val_loss,
            "processed_tokens": self.processed_tokens,
            "elapsed_seconds": time.monotonic() - self.started_at,
            "last_metrics": last_metrics,
        }
