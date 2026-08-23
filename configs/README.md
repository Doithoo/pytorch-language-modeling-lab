# Configuration Directory

The YAML files are executable examples, not hidden defaults:

| File | Purpose |
| --- | --- |
| `bigram_tiny.yaml` | Character bigram baseline |
| `char_tiny.yaml` | Character Transformer and dry run |
| `learning_cpu.yaml` | Complete small CPU training and evaluation |
| `bpe_tiny.yaml` | Byte-level BPE comparison |
| `kaggle_char_small.yaml` | Readable single-GPU training |
| `kaggle_char_small_sdpa.yaml` | SDPA attention comparison |
| `kaggle_char_small_rope.yaml` | RoPE positional encoding |
| `kaggle_char_small_rope_cached.yaml` | Shared RoPE cache optimization |
| `*_seed7.yaml`, `*_seed123.yaml` | Paired multiseed study configurations |

Use a separate dataset and run name when changing tokenizer or experiment identity. See the [configuration reference](../docs/reference/config-reference.md) and the [experiments guide](../docs/guides/experiments.md).
