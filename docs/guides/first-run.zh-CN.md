# 第一次本地运行

[教程首页](../tutorial/README.zh-CN.md)

## 1. 安装

```bash
uv sync --locked --extra dev
uv run lm --version
uv run lm list-models
uv run lm show-config --config configs/char_tiny.yaml
```

支持 Python 3.10-3.12。本地不要求 CUDA。

## 2. 准备语料

```bash
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
```

下载器校验 Tiny Shakespeare 的固定 SHA-256。准备阶段规范化文本、去重、按文档切分、只用 train split 建立词表，再写入 `.bin` token 文件和 manifest。

再次运行不会静默覆盖数据。确认要重建时使用：

```bash
uv run lm prepare-data --config configs/char_tiny.yaml --overwrite
```

## 3. 检查和 dry run

```bash
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

检查输出应包含 split 文档数、token 数、窗口数、manifest identity、batch shape、loss 和模型参数量。dry run 不创建 `artifacts/<run>`。

## 4. 下一步

完整训练使用 [Kaggle 指南](kaggle.zh-CN.md)。想先理解代码时阅读[代码导览](../concepts/code-tour.zh-CN.md)。
