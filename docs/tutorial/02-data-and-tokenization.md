# Data and Tokenization

The data pipeline is deliberately ordered:

```text
source validation -> normalization -> document split
-> tokenizer fit on train -> encode -> document-aware windows
-> manifest and hashes
```

A `.txt` file becomes one or more documents. A `.jsonl` file must contain one object per line with a string `text` field and may provide a unique string-like `id`.

The character tokenizer is deterministic. Byte-level BPE is trained with Hugging Face `tokenizers` and must use a separate `dataset_name` so its manifest and token files cannot be confused with the character experiment.

To prepare the BPE comparison:

```bash
uv run lm prepare-data --config configs/bpe_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus_bpe/manifest.json
```

A manifest identity changes when source bytes, normalization, split seed, tokenizer, context length, stride, or window policy changes. This identity is checked before training and evaluation.
