# 教程

本项目按照一组明确的输入、输出和文件规则介绍因果语言模型：

```text
download -> prepare -> inspect -> dry-run -> train -> evaluate -> generate
```

建议顺序：

1. [基础](00-basics.md)：bigram、右移 target 和 causal 约束。
2. [环境与数据](01-environment.md)：安装、下载和 manifest。
3. [数据与 tokenizer](02-data-and-tokenization.md)：字符级与 BPE。
4. [模型](03-models.md)：attention、Transformer 和 cache 规则。
5. [训练与恢复](04-training-and-resume.md)：验证、checkpoint 和 resume。
6. [评估与生成](05-evaluation-and-generation.md)：指标、采样和 KV cache。

最短本地流程：

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json --dry-run --device cpu
```

dry run 只检查一个 forward/backward batch，不代表模型质量。正式训练记录见 [recorded run](../recorded-run/README.zh-CN.md)。
