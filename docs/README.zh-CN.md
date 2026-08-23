# 文档首页

[English](README.md) | [项目首页](https://github.com/Doithoo/pytorch-language-modeling-lab/blob/main/README.zh-CN.md)

按目标选择页面。教程按顺序阅读；指南用于执行工作流；参考页定义精确的配置、输出和兼容性契约。

## 我想跑通项目

1. [教程首页](tutorial/README.zh-CN.md)：从基础到评估与生成。
2. [第一次运行](guides/first-run.zh-CN.md)：安装、下载、准备、检查和 dry run。
3. [使用自己的数据](guides/using-your-data.zh-CN.md)：`.txt`/`.jsonl` 和 manifest 边界。
4. [Kaggle 训练](guides/kaggle.zh-CN.md)：构建和提交 GPU runner。
5. [使用已发布模型](guides/using-models.zh-CN.md)：下载、校验、生成和评估。
6. [排障](guides/troubleshooting.zh-CN.md)：数据、设备、显存和续训问题。

## 我想理解原理

- [语言模型流程](concepts/language-modeling-flow.zh-CN.md)
- [Decoder-only Transformer](concepts/decoder-only-transformer.zh-CN.md)
- [生成与采样](concepts/generation.zh-CN.md)
- [代码导览](concepts/code-tour.zh-CN.md)
- [可复现实验契约](architecture/0001-reproducible-causal-language-model-contracts.zh-CN.md)

## 我需要查具体答案

| 问题 | 页面 |
| --- | --- |
| 应该运行哪个 CLI 命令？ | [CLI 与输出](reference/cli-and-outputs.zh-CN.md) |
| 配置字段是什么意思？ | [配置参考](reference/config-reference.zh-CN.md) |
| manifest 和 hash 如何工作？ | [数据 manifest](reference/data-manifest.zh-CN.md) |
| 语料来源和限制是什么？ | [数据集说明](reference/dataset-card.zh-CN.md) |
| perplexity、BPC 怎么读？ | [指标](reference/metrics.zh-CN.md) |
| checkpoint 中保存什么？ | [Checkpoint 结构](reference/checkpoint-schema.zh-CN.md) |
| 如何选择模型和配置？ | [模型清单](reference/model-zoo.zh-CN.md) |
| 如何设计实验比较？ | [实验比较](guides/experiments.zh-CN.md) |
| 如何添加模型？ | [添加模型](guides/adding-models.zh-CN.md) |

当前 `0.2.2` 版本验证见 [v3 运行记录](recorded-run/v3/README.zh-CN.md)。v2 仍保留为 readable、SDPA、RoPE 和多 seed 对照研究的历史记录。
