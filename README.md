# PyTorch Language Modeling Lab

[中文文档](README.zh-CN.md)

An educational, reproducible PyTorch project that teaches causal language modeling from a bigram baseline to a small GPT-style decoder-only Transformer.

Kaggle's free GPU environment is the primary environment for complete training and published runs. Local CPU execution covers data inspection, dry runs, and tests, so CUDA is not required on the learner's machine.

## Features

- Deterministic character tokenizer and byte-level BPE.
- Document-level train/validation/test splitting with leak-resistant manifests.
- Optional character-balanced splitting and source-group isolation.
- Memory-mapped token windows with shifted next-token targets.
- Bigram baseline and readable pre-LN causal Transformer.
- AdamW, warmup/cosine decay, AMP, gradient accumulation, and clipping.
- Versioned `best.pt`/`last.pt` checkpoints with optimizer, scheduler, scaler, tokenizer, and RNG state.
- Token-weighted loss, perplexity, top-1/top-5 accuracy, bits-per-character/byte, configurable prompt suites, and KV caching.
- Deterministic, resumable batch ordering with exact interrupted-vs-continuous training tests.
- Relocatable manifests, atomic dataset publication, strict configuration validation, and run comparison.
- A self-contained Kaggle runner with pinned Tiny Shakespeare SHA-256.

## Recorded Kaggle Run

version 0.2.0 of the project has completed a Tesla T4 recorded run using checkpoint schema v2, deterministic resume ordering, and token-aware evaluation. The best validation perplexity was **3.793904** and test perplexity **3.827989**. See the [latest recorded run](docs/recorded-run/v2/README.md) for metrics, environment, hashes, and fixed-prompt output.

## Kaggle Quick Start

```bash
uv sync --locked --extra dev
uv run python -m pytest -W error::DeprecationWarning
uv run python scripts/kaggle/build_snapshot.py
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

Change the account name in `docs/recorded-run/kaggle/kernel-metadata.json` before submitting. Keep GPU and Internet enabled. Completed evidence is published in `docs/recorded-run/`; dry runs and interrupted tasks are not reported as benchmark results.


## Local Smoke Run

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

See the [latest v2 recorded run](docs/recorded-run/v2/README.md) and the [Chinese project guide](README.zh-CN.md) for the complete workflow.

## Build a Comparable Experiment Report

After training two completed runs, compare their token-aware metrics:

```bash
uv run lm compare-runs --run artifacts/char-run --run artifacts/bpe-run
```

`test_perplexity` must only be compared when the dataset split and tokenizer are the same. For different tokenizers, use bits-per-character/byte and the qualitative prompt suite as additional context.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest -W error::DeprecationWarning
uv build
```

Tests use synthetic corpora and require neither a network connection nor a GPU. Licensed under the [MIT License](LICENSE).

