# Environment and Data

The repository supports Python 3.10-3.12 and uses `uv` for locked environments. CUDA is not required for tests or local dry runs.

```bash
uv sync --locked --extra dev
uv run lm --version
uv run lm list-models
```

Download the pinned corpus and prepare a relocatable manifest:

```bash
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
```

Preparation normalizes source text, splits documents, trains the tokenizer on the train split only, writes `uint32` token files, and records hashes in the manifest. It refuses to overwrite an existing dataset unless `--overwrite` is explicit.

The local CPU contract ends at inspection and a dry run. Use the [Kaggle guide](../guides/kaggle.md) for the complete small-model training workflow.
