# Model Zoo

| Configuration | What it teaches | Recommended environment |
| --- | --- | --- |
| `bigram_tiny.yaml` | Transition logits and sampling | CPU |
| `char_tiny.yaml` | Character Transformer and dry run | CPU or Kaggle |
| `bpe_tiny.yaml` | Byte-level BPE trade-offs | CPU preparation or Kaggle |
| `kaggle_char_small.yaml` | Complete readable training | Kaggle GPU |
| `kaggle_char_small_sdpa.yaml` | SDPA implementation comparison | Kaggle GPU |
| `kaggle_char_small_rope.yaml` | RoPE positional encoding | Kaggle GPU |
| `kaggle_char_small_rope_cached.yaml` | Shared RoPE cache optimization | Kaggle GPU |

The default route is `bigram_tiny` -> `char_tiny` -> `kaggle_char_small`. Use the recorded runs to compare attention and positional-encoding choices.
