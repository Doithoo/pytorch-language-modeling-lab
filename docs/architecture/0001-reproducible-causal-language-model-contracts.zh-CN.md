# ADR 0001：可复现因果语言模型契约

- 状态：已接受并由首个完整版本验证
- 范围：数据、tokenizer、decoder-only 模型、训练、评估、生成与 Kaggle 运行编排

## 背景

自回归语言模型的最终输出是一段文本，但影响结果的状态远不止模型权重：语料内容、文档划分、规范化规则、tokenizer 词表、上下文长度、训练随机状态和生成参数都会改变结果。如果这些状态没有被记录，实验无法比较，checkpoint 也不能可靠恢复。

## 决策

### 1. 使用 next-token prediction 作为共同训练契约

模型接收形状为 `[batch, time]` 的 `input_ids`，输出形状为 `[batch, time, vocab]` 的 logits。目标是输入右移一位后的 token 序列。loss 只在有效 target token 上计算，padding 使用明确的 `ignore_index`。

### 2. 先实现 decoder-only，再扩展其他架构

首期模型是因果 decoder。它直接对应文本续写、困惑度和生成，输入输出规则也较简单。encoder-only、seq2seq 和大规模预训练不进入首个版本。

### 3. tokenizer 是 checkpoint 的一等依赖

checkpoint 保存 tokenizer fingerprint、词表大小和特殊 token id。加载时若任一信息不匹配则拒绝运行，不能静默使用当前目录中的 tokenizer。

### 4. manifest 定义数据身份

manifest 保存源文件哈希、规范化方式、split、token 数、窗口参数和 tokenizer fingerprint。训练、评估和续训都验证 manifest identity。原始大文件不提交到仓库。

### 5. checkpoint 不保存可执行对象

模型通过注册表和可序列化配置重建，checkpoint 保存 state dict 而不是 Python 模型实例。外部模型工厂若未来开放，必须使用显式 `module:function` 路径，并把依赖版本作为实验元数据。

### 6. best 只由验证集选择

`best.pt` 按验证集 `val_loss` 最小选择，测试集只在实验决策冻结后执行一次。固定 prompt 生成样例用于定性观察，不作为唯一模型选择依据。

### 7. 清晰实现优先于优化实现

先提供可读的 attention 和无 cache 生成路径，再用测试和基准验证 KV cache、fused attention 或其他优化。优化实现必须与参考实现比较输出和性能。

### 8. Kaggle 是正式训练环境

Kaggle GPU 作为主训练环境。仓库代码通过可哈希的源码快照提交到 Kaggle kernel，runner 在 `/kaggle/working` 中执行完整流程并输出结构化运行摘要。本地环境只保证 CPU smoke test、单元测试和配置检查，不将本地运行作为正式 benchmark。

默认面向单张 GPU 的 `cuda:0` 设计，不假设 Kaggle 分配的多张 GPU 会被使用。任务开始必须记录 Python、PyTorch、CUDA、GPU 名称与显存、数据集版本、源码快照和配置哈希；设备不兼容时由 preflight 提前失败。

Kaggle 的 GPU、会话时长、磁盘和网络状态是外部运行条件，不能被 checkpoint 假定为永久存在。因此每个阶段必须保存 `last.pt`，最终结果只从下载并校验过的 artifacts 发布。

## 结果

该契约会使数据准备和 checkpoint 变得比最小 demo 更复杂，但可以把一次生成结果追溯到明确的语料、词表、配置和代码环境。它允许 bigram 与 Transformer 共用数据、训练和评估边界，而不把实现锁定在单一模型上。

## 非目标

该决策不承诺跨硬件逐位一致，不定义大规模分布式训练协议，也不保证生成文本具有事实性、无偏性或生产可用性。