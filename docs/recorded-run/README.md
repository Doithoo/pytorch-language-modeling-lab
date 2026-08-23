# Recorded Run Index

The machine-readable catalog is [`index.json`](index.json). It lists the historical runs and the current `0.2.2` release validation; no new model benchmark was run for the documentation and contract changes alone.

| Run | Release | Purpose | Seeds | Details |
| --- | --- | --- | --- | --- |
| Readable baseline | 0.2.0 | Readable reference implementation | 42 | [v2 report](v2/README.md) |
| Learned + SDPA | 0.2.0 | Paired control | 42, 7, 123 | [multiseed report](v2/multiseed/README.zh-CN.md) |
| RoPE + SDPA | 0.2.0 | Positional study | 42, 7, 123 | [multiseed report](v2/multiseed/README.zh-CN.md) |
| Cached RoPE | 0.2.1 | Cache optimization | 42 | [cache report](v2/rope-cache/README.zh-CN.md) |

The run records below use fixed Tiny Shakespeare configurations. They are reference runs, not universal benchmarks. Compare token-level perplexity only within the same manifest and tokenizer contract.
