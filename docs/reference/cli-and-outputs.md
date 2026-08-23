# CLI and Outputs

All commands are available through the `lm` entry point. Use `uv run lm --help` and `uv run lm <command> --help` for the authoritative option list.

| Command | Required input | Main output |
| --- | --- | --- |
| `prepare-data` | config | manifest directory and token `.bin` files |
| `inspect-data` | manifest | verified JSON summary |
| `train` | config, optional manifest | run directory, checkpoints, metrics |
| `evaluate` | checkpoint and manifest | JSON metrics |
| `generate` | checkpoint and prompt | JSON prompt, text, ids, parameters |
| `compare-runs` | two or more run directories | comparison JSON and safe rankings |
| `list-models` | none | model names and descriptions |
| `show-config` | config | fully resolved JSON config |
| `preflight` | optional work directory | environment and device information |

A completed run contains `config.yaml`, `manifest.json`, `environment.json`, `run.json`, `metrics.csv`, `tokenizer/`, `checkpoints/`, and `evaluation/`. `best.pt` is selected by validation loss; `last.pt` is the resumable training state.

Use `--output path.json` with `evaluate`, `generate`, and `compare-runs` to persist machine-readable results. The comparison output includes manifest/tokenizer identities and marks token-level perplexity as non-comparable when contracts differ.
