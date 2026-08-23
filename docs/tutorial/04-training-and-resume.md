# Training and Recovery

Run a CPU dry run before a full experiment:

```bash
uv run lm train \
  --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

A completed run writes `config.yaml`, `manifest.json`, `environment.json`, `run.json`, `metrics.csv`, `checkpoints/best.pt`, `checkpoints/last.pt`, and an `evaluation/` directory.

`best.pt` is selected by validation loss and is used for final evaluation. `last.pt` contains the most recent optimizer, scheduler, scaler, sampler, and RNG state and is the resume input.

Resume with the same configuration and an explicit checkpoint:

```bash
uv run lm train \
  --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --resume artifacts/char-tiny-transformer/checkpoints/last.pt
```

The trainer validates model, training, manifest, tokenizer, and data-order contracts before restoring state. Interrupted versus continuous training is tested on a synthetic corpus.
