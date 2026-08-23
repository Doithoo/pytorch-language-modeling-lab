# Using Your Data

Supported source files are UTF-8 `.txt` and `.jsonl` files under `data.corpus_dir`.

A JSONL row must contain a string `text` field and may contain a unique `id`:

```json
{"id": "doc-001", "text": "A short document."}
```

Text files are split into paragraphs or non-empty lines. Documents are assigned to train, valid, and test before tokenizer training. Use `group_by_source: true` when documents from one source must remain in one split.

Create a separate manifest for each tokenizer or context policy:

```bash
uv run lm prepare-data --config configs/bpe_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus_bpe/manifest.json
```

Do not edit `.bin`, tokenizer files, or manifests manually. Rebuild with `prepare-data --overwrite` when source text, normalization, split settings, tokenizer, context length, or stride changes.

See the [data manifest reference](../reference/data-manifest.md) for the identity and hash rules.
