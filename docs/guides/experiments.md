# Experiments

A fair comparison changes one meaningful variable at a time. Keep source files, split seed, manifest, tokenizer, model size, optimizer, update budget, and generation prompts fixed unless the experiment is explicitly about that variable.

Use separate dataset names and run names:

```bash
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --set runtime.run_name=char-run --device cpu
uv run lm train --config configs/bpe_tiny.yaml \
  --manifest data/manifests/tiny_corpus_bpe/manifest.json \
  --set runtime.run_name=bpe-run --device cpu
uv run lm compare-runs --run artifacts/char-run --run artifacts/bpe-run
```

The comparison command refuses to rank token-level perplexity when manifest or tokenizer identities differ. It reports bits-per-character and bits-per-byte for cross-tokenizer context. Report means and sample standard deviations for multi-seed studies; do not present a single Tiny Shakespeare run as a universal architecture claim.

Keep completed run directories immutable. Compare `best.pt` for final quality and `last.pt` only for recovery.
