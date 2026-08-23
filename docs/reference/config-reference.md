# Configuration Reference

Configuration is loaded from YAML and then repeated `--set section.field=value` overrides. Unknown fields fail early.

| Section | Important fields | Constraints |
| --- | --- | --- |
| `data` | corpus, processed and manifest directories; dataset name; context; stride; split; source URL/version/license | stride is 1..context; ratios are three positive values summing to 1 |
| `tokenizer` | `char` or `bpe`; vocab size; frequency; special tokens | BPE vocabulary is at least 32; four special tokens are unique |
| `model` | model name; layers; heads; width; FFN; positions; attention implementation | width divides heads; RoPE head dimension is even |
| `train` | updates; batch; accumulation; optimizer; AMP; eval/save intervals | positive intervals; warmup is at most max steps |
| `generation` | length; temperature; top-k; top-p; cache; seed; prompts | top-p is in `(0, 1]`; prompts are non-empty |
| `runtime` | device; seed; output directory; run name; deterministic | device is `auto`, `cpu`, `cuda`, or `mps` |

`model.vocab_size: null` is resolved from the prepared tokenizer. `model.context_length` must equal `data.context_length`. Optional `data.source_url`, `source_version`, and `source_license` are persisted in newly generated manifests. See the [configuration examples on GitHub](https://github.com/Doithoo/pytorch-language-modeling-lab/tree/main/configs) and use `lm show-config` to inspect defaults.
