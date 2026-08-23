# Seed 7 RoPE

Controlled replication of the RoPE + SDPA experiment using `seed=7` and the same 5,000-step budget.

```bash
uv run python scripts/kaggle/build_snapshot.py --config configs/kaggle_char_small_rope_seed7.yaml --output docs/recorded-run/kaggle-seed7-rope/run_kaggle.py
kaggle kernels push -p docs/recorded-run/kaggle-seed7-rope
```
