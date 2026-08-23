# Cached RoPE runner

This controlled run uses the seed-42 RoPE configuration with `0.2.1` precomputed non-persistent cos/sin buffers. It compares throughput and memory against the frozen dynamic-RoPE kernel while preserving checkpoint state-dict compatibility.

```bash
uv run python scripts/kaggle/build_snapshot.py \
  --config configs/kaggle_char_small_rope_cached.yaml \
  --output docs/recorded-run/kaggle-rope-cached/run_kaggle.py
kaggle kernels push -p docs/recorded-run/kaggle-rope-cached
```
