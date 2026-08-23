# First Run

Install the locked development environment:

```bash
uv sync --locked --extra dev
uv run lm --version
uv run lm list-models
uv run lm show-config --config configs/char_tiny.yaml
```

Download and prepare the pinned corpus:

```bash
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
```

Run one CPU forward/backward batch:

```bash
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

Preparation refuses to overwrite an existing dataset. Use `--overwrite` only when intentionally rebuilding it. A formal training run should use a new `runtime.run_name`.

A complete small CPU run is available after preparation:

```bash
uv run lm train --config configs/learning_cpu.yaml \
  --manifest data/manifests/tiny_corpus_learning_cpu/manifest.json \
  --device cpu
```

Continue with the [tutorial](../tutorial/README.md), [published model guide](using-models.md), or [Kaggle guide](kaggle.md).
