# Contributing

This is a small, reproducible causal language-modeling project. Keep contributions focused on data, tokenizers, models, training, evaluation, inference, and run documentation.

Before opening a pull request:

```bash
uv sync --locked --extra dev
make check
pre-commit run --all-files
```

Add focused tests for behavior changes. Integration tests must remain offline, CPU-only, and based on synthetic corpora. Do not commit raw corpora, generated token files, credentials, or model checkpoints. A new recorded run must include its exact source snapshot, data identity, environment, configuration, metrics, and checkpoint hashes.

When changing a public contract, update the relevant English and Chinese reference page and add a migration note if checkpoint or manifest compatibility changes.
