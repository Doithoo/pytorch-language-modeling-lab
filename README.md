# PyTorch Language Modeling Lab

[中文文档](README.zh-CN.md)

An educational, reproducible PyTorch project that teaches causal language modeling from a bigram baseline to a small GPT-style decoder-only Transformer.

Kaggle's free GPU environment is the primary environment for complete training and published runs. Local CPU execution covers data inspection, dry runs, and tests, so CUDA is not required on the learner's machine.

## Features

- Deterministic character tokenizer and byte-level BPE.
- Document-level train/validation/test splitting with leak-resistant manifests.
- Memory-mapped token windows with shifted next-token targets.
- Bigram baseline and readable pre-LN causal Transformer.
- AdamW, warmup/cosine decay, AMP, gradient accumulation, and clipping.
- Versioned `best.pt`/`last.pt` checkpoints with optimizer, scheduler, scaler, tokenizer, and RNG state.
- Token-weighted loss, perplexity, fixed-prompt samples, top-k/top-p generation, and KV caching.
- A self-contained Kaggle runner with pinned Tiny Shakespeare SHA-256.

## Recorded Kaggle Run

A completed Tesla T4 run trained the 10,763,136-parameter character model for 5,000 updates in 835.2 seconds. The step-3000 best checkpoint reached validation perplexity **3.786090** and test perplexity **3.815852** while peaking at 2.59 GiB CUDA memory. See the [recorded run](docs/recorded-run/README.zh-CN.md) for exact metrics, environment, hashes, and fixed-prompt output.

## Kaggle Quick Start

```bash
uv sync --locked --extra dev
uv run pytest -W error::DeprecationWarning
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

See [the Chinese project guide](README.zh-CN.md) and [Kaggle guide](docs/guides/kaggle.zh-CN.md) for the complete workflow.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
```

Tests use synthetic corpora and require neither a network connection nor a GPU. Licensed under the [MIT License](LICENSE).
