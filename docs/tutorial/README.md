# Tutorial Overview

This project teaches causal language modeling through small, inspectable contracts. Follow the pages in order when learning; use the guides and reference pages when operating the repository.

```text
download -> prepare -> inspect -> dry-run -> train -> evaluate -> generate
```

The first five tutorial pages use a pinned Tiny Shakespeare corpus and a character tokenizer. The project also includes a bigram baseline, byte-level BPE, learned positions, RoPE, readable attention, SDPA, checkpoint recovery, and KV-cache generation.

The shortest local path is:

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

A dry run checks one forward/backward batch. It is not a quality benchmark. Complete GPU results are documented in [recorded runs](../recorded-run/README.md).
