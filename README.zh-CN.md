# PyTorch 语言模型实验室

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-language-modeling-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**English: [README.md](README.md)**

这是一个面向学习者的 PyTorch 自回归语言模型项目。它从可读的 bigram 基线开始，逐步进入字符级/BPE tokenizer、GPT 风格 decoder-only Transformer、可恢复训练、困惑度评估和带 KV cache 的文本生成。

这是 `0.2.1` 优化版本。Kaggle 已完成 readable、SDPA、RoPE、seed 7/42/123 配对复现和 RoPE cache 性能验证。RoPE + SDPA 三 seed 平均 test perplexity 为 **3.610572**，learned + SDPA 为 3.859203；共享 autocast-aware cache 保持等价质量与基线级吞吐。最新记录见 [v2 recorded run](docs/recorded-run/v2/README.zh-CN.md)。

## 已完成的 Kaggle 训练

| 项目 | 结果 |
| --- | ---: |
| 模型 | 6 层、10,763,136 参数 decoder-only Transformer |
| 训练 | 5,000 updates，39,229,485 个有效 target tokens |
| 最佳验证 perplexity | **3.793904** |
| 测试 perplexity | **3.827989** |
| 测试 top-1 accuracy | **0.586632** |
| 测试 top-5 accuracy | **0.870946** |
| 测试 bits-per-character | **1.950089** |

![Tiny Shakespeare 训练与验证 loss](docs/recorded-run/v2/training-curves.png))

完整配置、20 次验证记录、环境、生成样例和 hash 见 [v2 Kaggle 训练记录](docs/recorded-run/v2/README.zh-CN.md)。仓库不提交 checkpoint，可从对应 kernel version 3 下载。

## 在 Kaggle 开始

正式流程会下载并校验 1.1 MB 的 Tiny Shakespeare，准备数据、训练字符级 small Transformer、选择最佳 checkpoint、评估 test split 并生成固定 prompt 样例。

### 1. 本地检查并构建自包含 runner

```bash
git clone https://github.com/Doithoo/pytorch-language-modeling-lab.git
cd pytorch-language-modeling-lab
uv sync --locked --extra dev
uv run python -m pytest -W error::DeprecationWarning
uv run python scripts/kaggle/build_snapshot.py
```

构建器把 `src/`、Kaggle 配置和包元数据压缩并嵌入 `run_kaggle.py`。Kaggle 非交互任务不依赖本地父目录或临时挂载的源码文件。

### 2. 登录并提交

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <你的用户名>/pytorch-language-modeling-lab-gpu
```

如果 Kaggle 用户名不是 `yashowhoo`，先修改 `docs/recorded-run/kaggle/kernel-metadata.json` 的 `id`。保持 GPU 和 Internet 开启；默认只使用 `cuda:0`。

### 3. 下载结果

```bash
kaggle kernels output <你的用户名>/pytorch-language-modeling-lab-gpu \
  --file-pattern 'artifacts/.*' \
  -p kaggle-output
```

重点查看 `metrics.csv`、`checkpoints/best.pt`、`checkpoints/last.pt`、`evaluation/summary.json`、`evaluation/prompts.jsonl`、`environment.json` 和 `kaggle-run-summary.json`。完整操作与中断续训见 [Kaggle 指南](docs/guides/kaggle.zh-CN.md)。

## 本地学习路径

### 安装和下载语料

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py
```

下载器固定官方来源和 SHA-256。接着准备字符级数据：

```bash
uv run lm prepare-data --config configs/char_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus/manifest.json
```

`prepare-data` 按文档切分 train/valid/test，只用 train split 建立词表，并生成带身份哈希的 `uint32` token 文件。若要重新准备，显式添加 `--overwrite`。

### CPU dry run

```bash
uv run lm train \
  --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

它只执行一个 batch 的 forward/backward，不创建正式运行目录。完整训练推荐提交 Kaggle。

### 评估和生成

```bash
uv run lm evaluate \
  --checkpoint artifacts/<run>/checkpoints/best.pt \
  --manifest data/manifests/<dataset>/manifest.json \
  --split test

uv run lm generate \
  --checkpoint artifacts/<run>/checkpoints/best.pt \
  --prompt "ROMEO:" --max-new-tokens 200
```

checkpoint 内嵌 tokenizer，因此生成不依赖当前目录中的隐式词表。评估仍要求 manifest，以确认 test 数据身份。若 manifest 与 token 文件不在原始绝对路径，可使用 `--processed-root` 指向 `.bin` 所在目录。

## 可以学到什么

- 连续文本如何变成右移一位的 `input_ids` 和 `targets`。
- 为什么必须先按文档切分，再训练 tokenizer 和创建窗口。
- 字符级 tokenizer 与 BPE 的词表、序列长度和未知字符差异。
- bigram 的 `[B,T,V]` logits 如何扩展到多层 causal Transformer。
- causal mask、multi-head attention、pre-LN、残差、GELU 和 embedding 权重绑定。
- 平均 token loss、perplexity、top-1/top-5 accuracy、bits-per-character/byte 和可配置 prompt 集。
- 由 `micro_step` 重建的确定性批顺序，支持中断后与连续训练逐张量一致。
- 可移动 manifest、双目录回滚发布、严格配置类型校验和运行对比 CLI。
- `best.pt` 与 `last.pt` 的区别，以及 optimizer/scheduler/RNG 如何恢复。
- temperature、top-k、top-p、EOS 与 KV cache 如何影响生成。
- 如何把 Kaggle 的 GPU、源码、语料、配置和 checkpoint 记录为可检查证据。

## 已实现的模型与 tokenizer

| 名称 | 用途 |
| --- | --- |
| `bigram` | 最小 next-token 基线，适合理解 logits、loss 和采样 |
| `transformer_decoder` | pre-LN GPT 风格 causal Transformer，支持 KV cache |
| `char` | 确定性字符词表，最适合第一条学习路线 |
| `bpe` | 基于 Hugging Face tokenizers 的 byte-level BPE |

配置入口：

- `configs/bigram_tiny.yaml`：bigram 基线。
- `configs/char_tiny.yaml`：本地字符级教学配置。
- `configs/bpe_tiny.yaml`：BPE 对照实验。
- `configs/kaggle_char_small.yaml`：Kaggle 正式训练起点。

## 项目结构

```text
configs/                     实验配置
data/                        raw、processed 与 manifests
artifacts/                   每次训练的隔离产物
docs/                        教程、概念、指南、参考和 recorded run
examples/                    短小教学代码
scripts/                     下载、绘图和 Kaggle runner 构建
src/language_model/
  data/                      语料、切分、manifest、memmap dataset
  tokenization/              char 与 BPE
  models/                    bigram、attention、Transformer、registry
  training/                  optimizer、trainer、checkpoint、orchestration
  evaluation/                loss、perplexity、checkpoint 评估
  generation/                采样、EOS、KV cache
  inference/                 checkpoint-backed Predictor
tests/                       unit、contract、integration
```

先从[文档首页](docs/README.zh-CN.md)选择学习路线。架构和兼容性边界见 [ADR 0001](docs/architecture/0001-reproducible-causal-language-model-contracts.zh-CN.md)。

## 开发检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
```

测试只使用合成语料，不联网、不下载模型、不要求 GPU。项目采用 [MIT License](LICENSE)。
