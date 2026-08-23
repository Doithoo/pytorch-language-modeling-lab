# Kaggle v3 训练记录

这是 `0.2.2` 工程改进后的完整 GPU 验证运行。项目使用 Kaggle kernel version 4 和 Tesla T4 提交了更新后的 runner。

- Kaggle：<https://www.kaggle.com/code/yashowhoo/pytorch-language-modeling-lab-gpu>
- Kernel version：4
- 版本：`0.2.2`
- 源码快照：`c9b552ba7752eb271670576da7b94160a563fa58594f7a7ba84bcd36e70c607c`，154,393 bytes
- Manifest identity：`22ab79d0c225b6b824e1f21a9a230036d0e1c880efeebd1a12975b71528553b3`
- Tokenizer fingerprint：`2f21f34c3e7146a043097c9d5c3e97a16de8d280f7e3a97fee031300ae5baf3e`

## 结果

| 指标 | 数值 |
| --- | ---: |
| 完成 updates | 5,000 |
| 最佳验证 step | 2,750 |
| 最佳验证 perplexity | **3.793904** |
| 测试 perplexity | **3.827989** |
| 测试 top-1 accuracy | 0.586632 |
| 测试 top-5 accuracy | 0.870946 |
| 有效训练 target tokens | 39,229,485 |
| 吞吐 | 42,466.1 token/s |
| 训练耗时 | 924.0 秒 |
| 峰值 CUDA 显存 | 2,829,815,296 bytes，约 2.64 GiB |

指标与历史 readable baseline 基本一致。本次运行用于确认 `0.2.2` 的源码快照、provenance 元数据、打包和文档发布，不代表质量提升。

## 环境

Python 3.12.13、PyTorch 2.13.0+cu130、CUDA 13.0、Tesla T4、compute capability 7.5。Kaggle 提供两张 GPU，项目只使用 `cuda:0`。

## 文件

- [`provenance.json`](provenance.json)：源码、语料、checkpoint、环境和结果 hash。
- [`data-summary.json`](data-summary.json)：轻量数据和 split 摘要。
- [`config.yaml`](config.yaml)：实际解析配置。
- [`metrics.csv`](metrics.csv)：全部验证记录。
- [`evaluation/summary.json`](evaluation/summary.json)：最终 valid/test 指标。
- [`evaluation/prompts.jsonl`](evaluation/prompts.jsonl)：固定 prompt 生成结果。
- [`runner-v4.py`](runner-v4.py)：实际提交到 Kaggle 的自包含 runner。

checkpoint 二进制不提交 Git。需要模型时从 kernel version 4 下载，并按 `provenance.json` 中的 SHA-256 校验。
