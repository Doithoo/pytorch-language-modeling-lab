# 在 Kaggle GPU 上训练语言模型

这是本项目推荐的正式训练方式。Kaggle 免费 GPU 用于完整语料训练、验证、最终评估和生成样例；本地环境只需要能够运行 CPU 测试和 dry run，不要求安装 CUDA。

> 已验证：`0.2.0` kernel version 3 已在 Tesla T4 上完成 5,000 updates、最终 valid/test 评估和 KV-cache 生成。实际结果见 [最新 v2 训练记录](../recorded-run/v2/README.zh-CN.md)。历史 version 2 记录仍保留用于追溯。

## 环境分工

| 环境 | 承担工作 |
| --- | --- |
| 本地 CPU | `ruff`、`mypy`、pytest、合成语料、配置检查、单 batch forward/backward |
| Kaggle GPU | 正式数据准备、small/medium 训练、验证、checkpoint、测试评估、固定 prompt 生成 |
| Kaggle 输出下载目录 | 检查 checkpoint、指标、环境记录和整理 recorded run |

项目默认使用单张 NVIDIA GPU，即 `cuda:0`。即使 Kaggle 页面显示两张 GPU，首期训练器也不会自动切换到分布式训练。

## Kaggle 目录约定

runner 启动后使用以下路径：

```text
/kaggle/input/                  # 只读 Kaggle Dataset 挂载
/kaggle/working/project/        # 源码快照
/kaggle/working/data/           # 下载或处理后的本次运行数据
/kaggle/working/artifacts/      # 需要从任务输出下载的正式产物
/kaggle/temp/                   # 可以丢弃的缓存和解压内容
```

训练代码不能把必须保留的 checkpoint 只写到 `/kaggle/temp`。原始语料和大体积 token cache 通常不应作为 kernel 输出发布。

## 数据提供方式

项目支持两种策略，具体语料确定后必须固定其中一种并写进 manifest。

### 方式 A：Kaggle Dataset 挂载

适合较稳定或下载较慢的语料：

1. 预先创建一个私有或公开 Kaggle Dataset。
2. 在 `kernel-metadata.json` 的 `dataset_sources` 中加入版本化数据源。
3. runner 从 `/kaggle/input/<dataset-slug>/` 只读访问。
4. `manifest.json` 记录 dataset owner、slug、version 和文件 SHA-256。

这种方式更稳定，可以关闭 Internet，但必须检查语料许可证是否允许重新发布。

### 方式 B：任务中从官方地址下载

适合体积小、地址稳定且许可证清晰的公开语料：

1. 在 metadata 中启用 Internet。
2. 只从文档声明的官方来源下载。
3. 下载后校验 SHA-256，再开始清洗和 tokenization。
4. 记录 URL、版本、许可证、下载时间和 hash。

非交互 Kaggle 任务不应在运行中临时创建或挂载新的 Kaggle Dataset。

## 提交前的本地检查

实现完成后，每次正式提交前运行：

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv run lm train --config configs/char_tiny.yaml --dry-run --device cpu
```

然后构建 runner 使用的源码快照。当前构建器会把 `src/`、正式配置、`pyproject.toml`、`uv.lock` 和由锁文件导出的 `requirements.lock` 放入自包含归档，并生成 SHA-256。不要让 runner 隐式读取未提交的父目录文件。

## 安装并登录 Kaggle CLI

Kaggle CLI 是提交工具，不应加入模型运行依赖：

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

如果网页已登录但 API 认证失败，可重新运行：

```bash
kaggle auth login --force
```

## 配置 kernel

打开 [`../recorded-run/kaggle/kernel-metadata.json`](../recorded-run/kaggle/kernel-metadata.json)，修改：

- `id` 中的 Kaggle 用户名和任务名。
- `dataset_sources`，如果使用 Kaggle Dataset。
- `enable_internet`，如果需要从官方 URL 下载。

保持 `enable_gpu: true` 和 `machine_shape: NvidiaTeslaT4`。当前 Kaggle PyTorch 不支持 P100 的 `sm_60`；runner 的 preflight 会打印并校验：

- Python、PyTorch 和 CUDA 版本。
- `torch.cuda.is_available()`。
- GPU 名称、compute capability、总显存和可用设备数。
- 工作目录剩余磁盘空间。
- 源码和配置 SHA-256。
- 语料或 Dataset 身份。

若 PyTorch 构建不支持分配到的 GPU，runner 应在数据下载和训练前失败。

## 提交与查看状态

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <你的用户名>/pytorch-language-modeling-lab-gpu
```

正常运行的结构化日志顺序为：

```text
preflight -> prepare_data -> inspect_data -> tokenizer
    -> smoke_batch -> train -> evaluate -> generate -> summarize
```

长训练每 60 秒输出一次心跳，至少包含 phase、global step、最近一次训练/验证 loss、已处理 token 数、耗时和显存峰值。心跳不代替 `metrics.csv`。

## 中断与续训

Kaggle 会话可能因配额、超时或平台状态结束，因此训练器必须：

- 按固定 step 保存 `last.pt`，写入时先临时保存再原子替换。
- 仅保留必要 checkpoint，避免耗尽磁盘。
- 在 checkpoint 中保存 optimizer、scheduler、AMP scaler、global step 和 RNG。
- 续训前校验配置、manifest 和 tokenizer fingerprint。

若任务中断，先下载已有输出，再把 `last.pt` 作为明确的输入上传或挂载到下一次任务；不要假设上一个 `/kaggle/working` 仍然存在。

## 下载产物

任务状态为 `COMPLETE` 后，只下载正式 artifacts：

```bash
kaggle kernels output <你的用户名>/pytorch-language-modeling-lab-gpu \
  --file-pattern 'artifacts/.*' \
  -p kaggle-output
```

重点检查：

| 文件 | 内容 |
| --- | --- |
| `environment.json` | Python、PyTorch、CUDA、GPU 和显存 |
| `run.json` | 配置、数据、源码和运行身份 |
| `metrics.csv` | step、loss、perplexity、学习率、吞吐和资源指标 |
| `checkpoints/best.pt` | 验证 loss 最小的模型 |
| `checkpoints/last.pt` | 最后训练状态，用于续训 |
| `evaluation/summary.json` | valid/test token loss 和 perplexity |
| `evaluation/prompts.jsonl` | 固定 prompt、生成参数和模型输出 |
| `kaggle-run-summary.json` | 总耗时、最佳 step、峰值显存和哈希摘要 |

## Recorded run 发布要求

项目只有在 Kaggle 任务真实完成后才更新 `docs/recorded-run/`。发布时保留：

- Kaggle 页面 URL 和完成日期。
- 实际模型参数、语料、tokenizer 和训练 token 数。
- GPU 型号、软件版本、总耗时和峰值显存。
- 最佳验证 perplexity 和一次冻结后的测试 perplexity。
- 源码快照、manifest 和被评估 checkpoint 的 SHA-256。
- 固定 prompt 生成样例和限制说明。

不把 dry run、被中断的任务或合成语料实验描述为完整训练结果。
