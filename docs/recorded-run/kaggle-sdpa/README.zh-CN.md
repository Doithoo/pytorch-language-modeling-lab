# SDPA 对照 runner

此目录提交与 readable attention 相同数据、配置和训练预算的 SDPA 对照实验。

```bash
uv run python scripts/kaggle/build_snapshot.py \
  --config configs/kaggle_char_small_sdpa.yaml \
  --output docs/recorded-run/kaggle-sdpa/run_kaggle.py
kaggle kernels push -p docs/recorded-run/kaggle-sdpa
```
