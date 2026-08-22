from __future__ import annotations

import json
from pathlib import Path

import pytest

from language_model.config import ExperimentConfig, config_from_dict


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    rows = [
        {"id": f"doc-{index}", "text": f"Document {index}: the quick brown fox learns language models. " * 3}
        for index in range(12)
    ]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return root


@pytest.fixture
def tiny_config(tmp_path: Path, corpus_dir: Path) -> ExperimentConfig:
    return config_from_dict(
        {
            "data": {
                "corpus_dir": str(corpus_dir),
                "processed_dir": str(tmp_path / "processed"),
                "manifest_dir": str(tmp_path / "manifests"),
                "dataset_name": "fixture",
                "tokenizer_name": "char",
                "context_length": 8,
                "stride": 8,
                "split_ratios": [0.7, 0.15, 0.15],
                "seed": 7,
            },
            "tokenizer": {"name": "char"},
            "model": {
                "name": "transformer_decoder",
                "context_length": 8,
                "n_layers": 1,
                "n_heads": 2,
                "d_model": 16,
                "d_ff": 32,
                "dropout": 0.0,
            },
            "train": {
                "max_steps": 2,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "eval_interval": 1,
                "eval_batches": 1,
                "save_interval": 1,
                "log_interval": 1,
            },
            "generation": {
                "max_new_tokens": 3,
                "temperature": 0.0,
                "top_k": None,
                "top_p": 1.0,
                "use_cache": True,
                "seed": 3,
            },
            "runtime": {
                "device": "cpu",
                "seed": 3,
                "output_dir": str(tmp_path / "artifacts"),
                "run_name": "fixture-run",
            },
        }
    )
