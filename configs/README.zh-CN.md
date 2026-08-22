# 配置目录

所有 YAML 都经过严格字段和跨字段校验。`model.vocab_size: null` 会在加载 manifest 后解析为真实词表大小。

| 文件 | 用途 |
| --- | --- |
| `bigram_tiny.yaml` | 字符级 bigram 教学基线 |
| `char_tiny.yaml` | 本地字符级 Transformer 与 CPU dry run |
| `bpe_tiny.yaml` | 512 词表 BPE 对照实验，使用独立 manifest |
| `kaggle_char_small.yaml` | `/kaggle/working` 路径、AMP、单 GPU 正式训练 |

先准备与配置完全对应的数据：

```bash
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm prepare-data --config configs/bpe_tiny.yaml
```

二者使用不同 `dataset_name`，不会共享 tokenizer 或 token 文件。修改 `context_length`、`stride`、split 或 tokenizer 后必须重新准备数据。

Kaggle small 默认 6 层、384 hidden、6 heads、256 context、batch 64、5,000 updates 和 AMP。这套配置已完成首次 T4 recorded run；它仍不是通用 benchmark。调整语料、GPU、显存或训练预算时，应发布实际解析配置和独立结果。
