# 配置目录

所有 YAML 都经过严格字段和跨字段校验。`model.vocab_size: null` 会在加载 manifest 后解析为真实词表大小。

| 文件 | 用途 |
| --- | --- |
| `bigram_tiny.yaml` | 字符级 bigram 教学基线 |
| `char_tiny.yaml` | 本地字符级 Transformer 与 CPU dry run |
| `bpe_tiny.yaml` | 512 词表 BPE 对照实验，使用独立 manifest |
| `kaggle_char_small.yaml` | `/kaggle/working` 路径、AMP、单 GPU、readable attention 正式训练 |
| `kaggle_char_small_sdpa.yaml` | 与 readable 相同预算的 PyTorch SDPA attention 对照 |
| `kaggle_char_small_rope.yaml` | SDPA + RoPE positional encoding 结构对照 |

先准备与配置完全对应的数据：

```bash
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm prepare-data --config configs/bpe_tiny.yaml
```

二者使用不同 `dataset_name`，不会共享 tokenizer 或 token 文件。新增 `kaggle_char_small_sdpa.yaml` 用于与 readable attention 做同预算对照。

Kaggle small 默认 6 层、384 hidden、6 heads、256 context、batch 64、5,000 updates 和 AMP。这套配置已分别完成历史 version 2 与 `0.2.0` version 3 recorded run；最新结果见 [`docs/recorded-run/v2/`](../docs/recorded-run/v2/README.zh-CN.md)。
