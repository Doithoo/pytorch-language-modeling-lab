# RoPE 对照 runner

此目录提交与 learned positional embedding 相同数据、seed、模型和训练预算的 RoPE 对照实验。

```bash
uv run python scripts/kaggle/build_snapshot.py \
  --config configs/kaggle_char_small_rope.yaml \
  --output docs/recorded-run/kaggle-rope/run_kaggle.py
kaggle kernels push -p docs/recorded-run/kaggle-rope
```
