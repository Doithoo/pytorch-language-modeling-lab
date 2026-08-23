# Documentation Index

[中文](README.zh-CN.md) | [Project home](https://github.com/Doithoo/pytorch-language-modeling-lab)

Choose a page by goal. The tutorial is executable and ordered; guides explain workflows; reference pages define exact contracts and output files.

## Run the project

1. [Tutorial](tutorial/README.md): basics through evaluation and generation.
2. [First run](guides/first-run.md): install, download, prepare, inspect, and dry-run.
3. [Using your data](guides/using-your-data.md): text/JSONL input and manifest boundaries.
4. [Kaggle training](guides/kaggle.md): build and submit the GPU runner.
5. [Using published models](guides/using-models.md): download, verify, generate, and evaluate.
6. [Troubleshooting](guides/troubleshooting.md): common data, device, memory, and resume failures.

## Understand the code

- [Language modeling flow](concepts/language-modeling-flow.md)
- [Decoder-only Transformer](concepts/decoder-only-transformer.md)
- [Generation and sampling](concepts/generation.md)
- [Code tour](concepts/code-tour.md)
- [Reproducible contracts](architecture/0001-reproducible-causal-language-model-contracts.md)

## Find an exact answer

| Question | Page |
| --- | --- |
| Which CLI command should I run? | [CLI and outputs](reference/cli-and-outputs.md) |
| What does a config field mean? | [Configuration](reference/config-reference.md) |
| What are manifests and hashes? | [Data manifest](reference/data-manifest.md) |
| How should I cite or assess the corpus? | [Dataset card](reference/dataset-card.md) |
| How do I read perplexity and BPC? | [Metrics](reference/metrics.md) |
| What is stored in a checkpoint? | [Checkpoint schema](reference/checkpoint-schema.md) |
| Which model/config should I use? | [Model zoo](reference/model-zoo.md) |
| How should experiments be compared? | [Experiments](guides/experiments.md) |
| How do I add a model? | [Adding models](guides/adding-models.md) |

The current `0.2.2` release validation is documented in the [v3 recorded run](recorded-run/v3/README.md). The earlier v2 package remains the canonical historical study for the readable, SDPA, RoPE, and multiseed comparisons.
