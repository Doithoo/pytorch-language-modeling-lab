# Evaluation and Generation

Evaluate a checkpoint against its matching prepared test split:

```bash
uv run lm evaluate \
  --checkpoint artifacts/<run>/checkpoints/best.pt \
  --manifest data/manifests/<dataset>/manifest.json \
  --split test --device cpu
```

Generate text using the tokenizer embedded in checkpoint schema v2:

```bash
uv run lm generate \
  --checkpoint artifacts/<run>/checkpoints/best.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 200 --device cpu
```

Generation supports greedy decoding (`--temperature 0`), temperature, top-k, top-p, EOS stopping, a seed, and KV cache. Use `--no-cache` to compare the reference full-forward path.

Perplexity is a token-level metric. When comparing char and BPE experiments, also report bits-per-character or bits-per-byte and keep the qualitative prompt suite fixed.
