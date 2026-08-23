# Seed 123 SDPA

Controlled replication of the learned-position SDPA baseline using `seed=123` and the same 5,000-step budget.

```bash
uv run python scripts/kaggle/build_snapshot.py --config configs/kaggle_char_small_sdpa_seed123.yaml --output docs/recorded-run/kaggle-seed123-sdpa/run_kaggle.py
kaggle kernels push -p docs/recorded-run/kaggle-seed123-sdpa
```
