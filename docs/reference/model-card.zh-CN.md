# Model Card：PyTorch 语言模型项目参考模型

## 概要

本项目提供小型因果语言模型：字符级 bigram 基线，以及支持 learned/RoPE 位置编码的 decoder-only Transformer。参考 checkpoint 使用 Tiny Shakespeare 训练，用于学习、回归测试和对照实验。

## 适用场景

- 学习 tokenizer、右移 target、causal attention、checkpoint 恢复和采样。
- 对比 readable attention、PyTorch SDPA、learned positions、RoPE 和 KV cache。
- 在本地 CPU 运行 smoke test，或在 Kaggle 完成小规模 GPU 实验。

## 非适用场景

这些模型不是生产对话系统、指令模型、事实知识库、安全分类器或通用语言模型。不应将其用于高风险决策、自动发布，或用于证明模型具有广泛能力。

## 数据与训练

公开研究使用[数据集说明](dataset-card.zh-CN.md)中的固定 Tiny Shakespeare 来源。数据先规范化和切分，再只使用 train split 训练 tokenizer。训练目标是 next-token cross-entropy；valid loss 用于选择 `best.pt`，最终选择后才计算 test 指标。

readable、SDPA、learned、RoPE 和共享 RoPE cache 的受控结果见[运行记录索引](../recorded-run/README.zh-CN.md)。

## 指标与限制

perplexity 是 token-level 指标，依赖 tokenizer。不同 tokenizer 应结合 BPC/BPB 判断。语料规模小、仅英文且具有文学体裁，结果不能外推到现代网页文本、多语言、长上下文、指令跟随、事实性或安全性。

生成文本只能作为定性参考，可能不连贯，受训练语料偏置影响，也不代表事实正确。

## 可复现性

每个公开实验记录源码快照、manifest identity、tokenizer fingerprint、环境、配置、指标和 checkpoint hash。checkpoint 不提交 Git，应从关联的 Kaggle 运行下载，并依据 provenance 校验。

## 许可证

项目代码使用 MIT License。再分发语料或派生产物前，应确认上游语料条款。
