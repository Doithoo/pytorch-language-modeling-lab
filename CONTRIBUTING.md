# Contributing

This is an educational, reproducible causal language-modeling lab. Keep contributions focused on the data, tokenizer, model, training, evaluation, inference, and documented experiment workflow.

Before opening a pull request:

```bash
uv sync --locked --extra dev
make check
pre-commit run --all-files
```

Add focused tests for behavior changes. Integration tests must remain offline, CPU-only, and based on synthetic corpora. Do not commit raw corpora, generated token files, credentials, or model checkpoints. A new recorded run must include its exact source snapshot, data identity, environment, configuration, metrics, and checkpoint hashes.

When changing a public contract, update the relevant English and Chinese reference page and add a migration note if checkpoint or manifest compatibility changes.
