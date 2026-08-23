# 环境与数据

项目支持 Python 3.10-3.12，使用 `uv` 管理锁定环境。本地测试和 dry run 不要求 CUDA。

```bash
uv sync --locked --extra dev
uv run lm --version
uv run lm list-models
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
```

准备阶段会规范化文本、切分文档、只用 train split 建立 tokenizer、写入 `uint32` token 文件并记录 manifest hash。已有数据不会被静默覆盖。

本地流程用于检查和 dry run；完整 small Transformer 训练使用 [Kaggle 指南](../guides/kaggle.zh-CN.md)。
