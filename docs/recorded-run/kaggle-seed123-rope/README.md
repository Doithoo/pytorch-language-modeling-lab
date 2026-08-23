# Seed 123 RoPE

Controlled replication of the RoPE + SDPA experiment using `seed=123` and the same 5,000-step budget.

```bash
uv run python scripts/kaggle/build_snapshot.py --config configs/kaggle_char_small_rope_seed123.yaml --output docs/recorded-run/kaggle-seed123-rope/run_kaggle.py
kaggle kernels push -p docs/recorded-run/kaggle-seed123-rope
```
