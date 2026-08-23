# PyTorch Language Modeling Project

[中文文档](README.zh-CN.md) | [Documentation](docs/README.md)

A small, reproducible PyTorch project for causal language modeling. It starts with a readable bigram baseline and covers character/BPE tokenization, a decoder-only Transformer, resumable training, perplexity evaluation, and KV-cache generation.

```text
download -> prepare -> inspect -> dry-run -> train -> evaluate -> generate
```

The project supports deterministic character tokenization, byte-level BPE, document-aware manifests, readable and SDPA attention, learned positions and RoPE, checkpoint recovery, token-weighted metrics, and KV-cache generation. Kaggle GPU is the formal environment for complete training; local CPU covers preparation, dry runs, tests, and small synthetic experiments.

## Results

The current release is `0.2.2`. A completed Kaggle GPU validation run is recorded in [v3](docs/recorded-run/v3/README.md). Its readable baseline metrics are consistent with the historical study; `0.2.2` is an engineering, documentation, provenance, and comparison-contract release, not a quality improvement claim.

The three-seed Tiny Shakespeare study reports mean test perplexity **3.610572** for RoPE + SDPA versus 3.859203 for learned + SDPA. The `0.2.1` shared RoPE cache retained equivalent quality. Historical comparisons are in the [v2 comparison records](docs/recorded-run/v2/README.md).

| Experiment | Release | Test perplexity | Purpose |
| --- | --- | ---: | --- |
| Readable attention | 0.2.0 | 3.827989 | Readable baseline |
| Learned + SDPA | 0.2.0 | 3.859203 mean | Attention implementation comparison |
| RoPE + SDPA | 0.2.0 | **3.610572 mean** | Three-seed positional study |
| Cached RoPE | 0.2.1 | 3.570488 (seed 42) | Cache optimization |

These are fixed-configuration runs on Tiny Shakespeare, not general language-model benchmarks. Token-level perplexity should not be ranked across different tokenizers without BPC/BPB context. See the [current v3 validation](docs/recorded-run/v3/README.md) and [historical v2 evidence](docs/recorded-run/v2/README.md).

## Local Smoke Run

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

The dry run performs one forward/backward batch and does not create a formal run directory. For a complete walkthrough, read the [tutorial](docs/tutorial/README.md). For your own corpus, read [Using your data](docs/guides/using-your-data.md).

## Kaggle Quick Start

```bash
uv sync --locked --extra dev
make test
uv run mkdocs build --strict
uv run python scripts/kaggle/build_snapshot.py
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

Change the account name in `docs/recorded-run/kaggle/kernel-metadata.json` before submitting. Keep GPU and Internet enabled. Completed evidence is published in `docs/recorded-run/`; dry runs and interrupted tasks are not benchmark results. See the [Kaggle guide](docs/guides/kaggle.md).

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
make test
uv run mkdocs build --strict
uv build
```

Tests use synthetic corpora and require neither a network connection nor a GPU. The project is MIT licensed. Read the [documentation index](docs/README.md), [contributing guide](CONTRIBUTING.md), and [security policy](SECURITY.md) before changing public contracts.
