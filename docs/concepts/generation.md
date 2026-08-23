# Generation and Sampling

The final-position logits are transformed by the selected sampling policy:

- `temperature=0` uses greedy argmax.
- Lower positive temperature concentrates probability.
- `top-k` keeps the highest-scoring candidates.
- `top-p` keeps the smallest candidate set reaching the probability threshold.
- EOS ends generation early.

The generator uses a local seeded `torch.Generator`. Exact output is expected for the same checkpoint, prompt, parameters, device, and seed, but cross-hardware bitwise identity is not promised.

KV cache reduces repeated prompt computation. `--no-cache` is the reference comparison path:

```bash
uv run lm generate --checkpoint best.pt --prompt "ROMEO:" --no-cache
```

Fluent text is not evidence of factuality, safety, or lack of bias. This repository trains a small continuation model for education.
