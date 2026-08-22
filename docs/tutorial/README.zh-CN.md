# 教程

[文档首页](../README.zh-CN.md) | [项目首页](../../README.zh-CN.md)

建议按下面顺序学习：

1. **环境与数据**：运行 `uv sync`，下载 Tiny Shakespeare，执行 `prepare-data` 和 `inspect-data`。
2. **bigram 基线**：使用 `configs/bigram_tiny.yaml` 理解 `[B,T,V]` logits、右移 target 和交叉熵。
3. **causal attention**：运行 `examples/02_causal_attention.py`，确认未来 token 不改变过去位置的 logits。
4. **Transformer dry run**：使用 `configs/char_tiny.yaml` 在 CPU 完成一个 forward/backward batch。
5. **Kaggle 训练**：构建自包含 runner，提交 `configs/kaggle_char_small.yaml`。
6. **评估与生成**：读取 `metrics.csv`、perplexity、固定 prompt 和 KV cache 输出。
7. **BPE 对照**：准备 `configs/bpe_tiny.yaml` 的独立 manifest，比较词表与序列长度。

最短本地流程：

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json --dry-run --device cpu
```

不要把 dry run 的 loss 当作模型质量。正式结果来自完整 Kaggle 训练，并使用验证集选出的 `best.pt` 一次性评估 test split。
