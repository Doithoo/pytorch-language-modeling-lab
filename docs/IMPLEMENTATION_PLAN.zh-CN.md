# 语言模型项目实施方案

> 状态：首个完整版本与 Kaggle recorded run 已完成。字符级/BPE、bigram/Transformer、训练恢复、评估、KV cache、CLI、测试、自包含 runner 和 T4 训练证据均已落地。

## 1. 目标与边界

### 1.1 首个完成目标

交付一个可以在小语料和 Kaggle 单张免费 GPU 上完成训练的 decoder-only Transformer，并且让读者能够从代码和实验产物中回答以下问题：

- 训练样本如何从连续语料变成 `input_ids` 和右移一位的 `target_ids`。
- 为什么注意力矩阵必须使用上三角 causal mask。
- 交叉熵、平均 token loss 与 perplexity 的关系是什么。
- tokenizer、语料划分、上下文长度和 checkpoint 为什么必须绑定。
- 生成时 temperature、top-k、top-p 如何改变输出分布。
- 如何从 `best.pt` 复现评估结果，如何从 `last.pt` 继续训练。

### 1.2 首期明确不做

首期不做大规模预训练、指令微调、RLHF、分布式并行、MoE、在线服务、长上下文优化和追求 SOTA 的 benchmark。它们会显著扩大工程面，且无法替代基础语言模型链路的正确性验证。

### 1.3 默认模型路线

```text
字符级 bigram
    -> 字符级 decoder-only Transformer
    -> 子词 tokenizer + decoder-only Transformer
    -> 生成优化：KV cache、批量生成、采样策略
    -> 可选研究扩展：RoPE、GQA、Flash Attention、分布式训练
```

默认模型使用 pre-LN decoder block、causal self-attention、GELU MLP、learned positional embedding 和 tied input/output embedding。每一步扩展都必须有独立配置和对照实验，避免一次加入多个变量。

### 1.4 训练环境策略

Kaggle GPU 是正式训练、性能测量和发布结果的主环境；本地机器不承担完整训练基准。环境边界固定如下：

| 场景 | 环境 | 目的 |
| --- | --- | --- |
| 单元与契约测试 | 本地 CPU | 快速验证配置、tokenizer、mask、loss、checkpoint 和采样 |
| tiny corpus 过拟合 | 本地 CPU 或 Kaggle CPU | 验证模型逻辑和数据链路，不作为性能结果 |
| small/medium 训练 | Kaggle 单张 GPU | 获取正式 loss、perplexity、吞吐、显存和耗时 |
| 最终评估与生成 | Kaggle GPU 或下载后的兼容环境 | 从固定 checkpoint 生成可发布证据 |

Kaggle runner 必须把源码、配置和必要脚本组成可标识的快照，在 `/kaggle/working/project` 中运行。默认使用 `cuda:0`，不依赖多卡；任务开始时记录 `torch.__version__`、CUDA 版本、GPU 名称、显存、Python 版本、磁盘空间和随机种子。GPU 具体型号以 Kaggle 当次分配为准，若当前 PyTorch 构建不支持该设备，应在 preflight 阶段直接失败并给出更换 GPU 的提示。

Kaggle 任务应开启 Internet 仅用于下载已声明来源的语料和依赖；更稳定的方案是预先准备 Kaggle Dataset，并把 dataset version 写入 manifest。无论采用直接下载还是 Dataset 挂载，都不能依赖用户本地未记录的文件。

本地 `--dry-run` 必须不下载大语料、不要求 CUDA，并能用合成数据验证完整命令编排。只有 Kaggle 运行才生成项目对外宣称的正式训练结果。

## 2. 工程分层

### 2.1 `src/language_model/config.py`

定义并校验 `DataConfig`、`TokenizerConfig`、`ModelConfig`、`TrainConfig`、`GenerationConfig` 和 `RuntimeConfig`。配置加载顺序固定为：代码默认值 -> YAML -> CLI dotted overrides。所有路径、数值范围、维度整除关系和互相依赖在训练开始前校验。

关键校验包括：

- `context_length >= 1`，且模型位置编码容量不小于它。
- `d_model % n_heads == 0`。
- `stride` 在 `[1, context_length]` 内。
- `vocab_size` 必须与 tokenizer manifest 一致，不能由用户手工猜测。
- `eval_interval`、`save_interval` 和 `max_steps` 为正整数。
- `pad_id`、`bos_id`、`eos_id`、`unk_id` 不重复。

### 2.2 `src/language_model/data/`

职责是语料生命周期，不负责模型逻辑：

- `sources.py`：读取纯文本、JSONL 等明确支持的源格式。
- `clean.py`：Unicode 规范化、空白处理、重复样本与异常行统计。
- `split.py`：按文档或记录切分，避免同一文档泄漏到多个 split。
- `manifest.py`：记录文件列表、字节哈希、样本数、token 数、split 和版本。
- `dataset.py`：从 token ids 产生固定长度窗口。
- `collate.py`：处理 padding、attention mask 和 batch 元数据。
- `inspection.py`：输出字符/token 频次、长度分布、空样本和泄漏检查。

首版优先支持本地 UTF-8 `.txt` 与 `.jsonl`，每条 JSONL 至少包含 `text` 字段。下载第三方语料的脚本必须记录来源、许可证、版本和校验值，原始大文件不进入 Git。

### 2.3 `src/language_model/tokenization/`

定义统一 tokenizer 接口：

```text
encode(text, add_bos, add_eos) -> list[int]
decode(ids, skip_special_tokens) -> str
vocab_size -> int
special_tokens -> mapping
fingerprint() -> str
save(path) / load(path)
```

先实现 deterministic `CharTokenizer`，再实现可训练的子词 tokenizer。tokenizer 输出必须包含版本和特殊 token 映射。训练集之外的 split 只能使用已经训练完成的 tokenizer，不能重新建立词表。

### 2.4 `src/language_model/models/`

- `spec.py`：统一模型元数据和输出契约。
- `bigram.py`：最小可读基线，输入 `[B, T]`，输出 `[B, T, V]` logits。
- `transformer.py`：decoder-only Transformer。
- `attention.py`：scaled dot-product attention、causal mask 和可选实现。
- `registry.py`：按名称构造内置模型，禁止在 checkpoint 中序列化可执行对象。
- `initialization.py`：权重初始化和参数统计。

训练前向接口约定为 `model(input_ids, targets=None) -> ModelOutput`。`logits` 是必需的，传入 targets 时返回 token loss；推理模式不能隐式修改模型或全局随机状态。

### 2.5 `src/language_model/training/`

- `losses.py`：next-token cross entropy，忽略 padding，报告有效 token 数。
- `optimizer.py`：AdamW 和 scheduler 构造。
- `trainer.py`：梯度累积、AMP、梯度裁剪、评估、日志和保存。
- `checkpoint.py`：保存与加载完整训练状态。
- `seed.py`：Python、NumPy、PyTorch 和 DataLoader worker 的随机状态。
- `logging.py`：JSONL/CSV 训练记录，避免只能从终端复制结果。

训练器必须区分 `global_step`、`micro_step` 和 epoch。验证 loss 不能参与梯度计算，测试集不用于选 best checkpoint。

### 2.6 `src/language_model/evaluation/` 与 `generation/`

评估至少输出：平均 token loss、perplexity、有效 token 数和每个 split 的数据身份。生成模块负责 temperature、top-k、top-p、greedy、随机种子、最大新 token 数和 EOS 停止条件。评估使用固定 prompt 集保存可比较的文本样例，但生成样例不能替代 perplexity。

### 2.7 `scripts/kaggle/` 与 CLI

Kaggle 脚本负责环境适配和运行编排，不承载模型逻辑：

- `build_snapshot.py`：生成带 SHA-256 的源码和配置快照。
- `run_kaggle.py`：在 Kaggle 中执行 preflight、prepare-data、train、evaluate 和 generate。
- `kernel-metadata.json`：记录账户占位符、GPU、Internet、入口脚本和 kernel 标识。
- `collect_output.py`：从 Kaggle 输出中提取 checkpoint、metrics、summary 和样例。

`src/language_model/inference/` 负责 checkpoint-backed 推理，CLI 负责本地和 Kaggle 内部的模块编排。CLI 计划支持：

```text
lm inspect-data
lm prepare-data
lm train [--dry-run]
lm evaluate
lm generate
lm list-models
lm show-config
```

## 3. 数据契约

### 3.1 数据流水线

```text
raw source
  -> source validation
  -> normalized documents
  -> document-level train/valid/test split
  -> tokenizer fit on train only
  -> encoded ids
  -> contiguous or document-aware windows
  -> manifest + statistics
```

默认使用文档级切分。连续语料窗口不能跨越文档边界，除非配置显式选择 packed-corpus 模式并记录该选择。训练样本采用：

```text
input_ids  = tokens[t : t + T]
target_ids = tokens[t + 1 : t + T + 1]
```

`target_ids` 的最后一个位置来自下一个 token；如果样本不足，使用 EOS 或丢弃，规则必须写进 manifest。

### 3.2 Manifest 最低字段

`schema_version`、`dataset_name`、`source_files`、`source_sha256`、`normalization`、`split_counts`、`token_counts`、`tokenizer_name`、`tokenizer_fingerprint`、`vocab_size`、`special_token_ids`、`context_length`、`stride`、`window_mode`、`created_at` 和 `manifest_sha256`。

任何影响 token 序列的变化都必须改变 manifest identity。训练、评估、续训和推理加载 checkpoint 时验证该 identity。

## 4. 模型实现顺序

### 阶段 A：bigram 基线

目标：用最少代码完成 token id -> logits -> loss -> 采样。先在内存 tiny corpus 上验证：loss 能下降、模型输出形状正确、生成只使用合法词表 id。该阶段不引入 Transformer 抽象。

验收：`test_bigram_shapes`、`test_next_token_loss`、`test_bigram_can_overfit_tiny_corpus` 和固定 seed 下的生成 smoke test。

### 阶段 B：最小 Transformer

实现单头注意力、可视化 causal mask，再扩展为多头注意力和 decoder block。顺序固定为：

1. token embedding 与位置 embedding。
2. `QKV` 投影与 attention score。
3. 上三角 mask，禁止看到未来 token。
4. attention dropout、residual、LayerNorm、GELU MLP。
5. 多层堆叠、最终 LayerNorm、lm head。
6. 输入/输出 embedding 权重绑定。

验收：Transformer 在 tiny corpus 上过拟合；未来 token 改变不会影响当前位置 logits；参数量和配置一致；CPU forward/backward 可完成。

### 阶段 C：数据与训练系统

先实现本地文本准备，再加入 manifest、窗口化 dataset、batch collator、验证集、checkpoint 和 resume。加入 AMP、梯度累积和梯度裁剪时，每次只打开一个配置开关并记录对照结果。

验收：`--dry-run` 不写训练产物；小规模训练产生 `config.yaml`、`manifest.json`、`metrics.csv`、`best.pt`、`last.pt` 和 `run.json`；中断后续训的 step、optimizer、scheduler 和 RNG 可以恢复。

### 阶段 D：子词 tokenizer

在字符级结果稳定后引入 BPE 或 unigram tokenizer。对相同语料和大致相同计算预算比较：词表大小、平均序列长度、训练 loss、验证 perplexity、生成速度和未知字符处理方式。

验收：tokenizer 可以独立保存/加载；checkpoint 拒绝加载不匹配词表；同一输入 encode/decode 在规定范围内稳定。

### 阶段 E：生成与推理

先实现 greedy 和 temperature，再加入 top-k、top-p、EOS、长度限制和随机种子。随后加入单条生成的 KV cache，并与无 cache 版本比较输出一致性和速度。缓存必须按 batch、层、头和序列长度明确管理。

验收：固定 seed 下生成可复现；`temperature=0` 的行为明确；top-k/top-p 不产生非法 id；cache 与 no-cache logits 在容差内一致；达到 EOS 后停止。

### 阶段 F：实验和发布

固定一套 tiny、small、medium 配置，记录硬件、总 token 数、训练步数、耗时、峰值显存、参数量、最佳验证 perplexity 和固定 prompt 输出。加入绘图脚本和一次完整 recorded run，再发布模型卡片、数据许可证说明和限制。

## 5. 测试计划

### 单元测试

- 配置未知字段、边界值和维度冲突会失败。
- tokenizer 的特殊 token、encode/decode、保存/加载和 fingerprint。
- 文档切分不泄漏，manifest 哈希稳定。
- 窗口输入和 target 正确右移一位。
- causal mask 的未来区域全部不可见。
- loss 只统计有效 token，padding 不影响结果。
- top-k、top-p、temperature 和 EOS 规则。

### 集成测试

- bigram 和 Transformer 都能在 tiny corpus 上过拟合。
- `prepare-data -> train --dry-run -> train -> evaluate -> generate` 串联成功。
- checkpoint 可在新进程中重建模型。
- 不匹配配置、tokenizer 或 manifest 时给出明确错误。

### 非功能检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
```

测试默认只使用合成语料和临时目录，不自动下载大型数据集、不依赖 GPU、不要求网络。Kaggle runner 另设一条 smoke path，在正式训练前验证 CUDA、GPU 设备、数据可见性、磁盘空间、源码快照和一个 forward/backward batch。

## 6. 产物规范

每次运行目录建议如下：

```text
artifacts/<run-name>/
├── config.yaml
├── run.json
├── kaggle-run-summary.json
├── environment.json
├── manifest.json
├── tokenizer/
├── metrics.csv
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── evaluation/
│   ├── summary.json
│   └── prompts.jsonl
└── samples/
    └── step-*.jsonl
```

`best.pt` 按验证集 `val_loss` 最小选择，`test` 只在最终模型确定后运行。checkpoint 至少保存：schema version、模型名称和超参、tokenizer fingerprint、manifest identity、模型状态、优化器和 scheduler 状态、global step、指标历史、设备/软件版本和 RNG 状态。

## 7. Kaggle 运行与产物收集

每次正式训练按以下顺序执行：

1. 本地运行 `ruff`、`mypy`、pytest 和 `lm train --dry-run`。
2. 构建源码快照，写入源码哈希、配置哈希和依赖约束。
3. 修改 `docs/recorded-run/kaggle/kernel-metadata.json` 中的 Kaggle 用户名和任务名，启用 GPU；根据语料来源决定是否启用 Internet。
4. 使用 `kaggle kernels push -p docs/recorded-run/kaggle` 提交，不在 notebook 中临时修改模型代码。
5. runner 在 Kaggle 中执行 preflight、数据准备、训练、验证、最终评估和固定 prompt 生成，并每隔一段时间输出结构化心跳。
6. 任务完成后只下载 `artifacts/` 和运行摘要，补充 `docs/recorded-run/README.zh-CN.md`、配置、指标和环境信息。

Kaggle 运行的最低完成证据是：任务状态成功、GPU 设备信息、训练/验证 token 数、训练步数、最佳验证 perplexity、最终测试 perplexity、总耗时、峰值显存、checkpoint SHA-256、源码快照 SHA-256 和固定 prompt 样例。Kaggle 临时目录中的原始语料、缓存和中间 checkpoint 不作为仓库发布内容。

## 8. 推荐实施节奏

1. 第 1 个工作单元：配置、CLI 骨架、数据 schema、synthetic corpus、preflight 和 Kaggle runner 骨架。
2. 第 2 个工作单元：CharTokenizer、窗口 dataset、manifest 和数据检查；先在 Kaggle 上验证数据路径。
3. 第 3 个工作单元：bigram、loss、最小训练循环和第一个本地/Kaggle 对照 smoke test。
4. 第 4 个工作单元：Transformer、causal mask、参数统计和 tiny corpus 过拟合。
5. 第 5 个工作单元：完整 trainer、验证、checkpoint、resume 和指标文件；完成一次 Kaggle small run。
6. 第 6 个工作单元：固定 prompt 评估、采样策略和可复现生成。
7. 第 7 个工作单元：子词 tokenizer、对照实验和 tokenizer 兼容性；在同一 Kaggle GPU 上比较。
8. 第 8 个工作单元：KV cache、性能测量、文档教程和正式 recorded run。

每个工作单元都应先补测试和契约，再写实现，最后更新教程和示例。只有上一阶段的验收条件满足后才进入下一阶段。

## 9. 主要风险与处理

- **Kaggle 运行中断**：每个阶段保存 `last.pt` 和结构化心跳；续训时先下载 checkpoint，验证 manifest/config identity 后再恢复。
- **Kaggle GPU/版本不兼容**：preflight 检查 CUDA、设备 capability 和 PyTorch 构建支持范围；不在训练中途才暴露设备问题。
- **Kaggle 磁盘或会话限制**：控制语料缓存、checkpoint 保留数量和日志频率；把最终产物与临时文件分离。
- **本地与 Kaggle 结果差异**：本地只做逻辑正确性，不比较绝对 loss；正式指标只引用固定 Kaggle 环境的 recorded run。
- **数据泄漏**：按文档切分并在 manifest 中保存样本身份；不对整段拼接文本随机切窗后再切分。
- **词表不一致**：词表单独版本化，checkpoint 强制校验 fingerprint。
- **看见未来信息**：mask、模型输出和 tiny corpus 行为都做测试。
- **只看生成样例**：使用 perplexity 和固定 prompt 双重报告，样例仅做定性补充。
- **训练不可续接**：保存 optimizer、scheduler、scaler 和 RNG，而不只保存模型权重。
- **过早性能优化**：先保留清晰实现，用 profiler 和基准数据决定是否引入 cache 或 fused kernel。
- **语料许可证不清晰**：数据下载脚本和 recorded run 同步保存来源、版本、许可证和 hash。
