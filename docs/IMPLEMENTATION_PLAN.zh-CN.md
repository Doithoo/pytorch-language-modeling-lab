# 当前架构与路线

> 状态：首个完整版本已完成；本文只描述当前真实存在的模块和已接受的契约。

## 当前流水线

```text
source -> normalize -> document split -> train tokenizer
       -> encode -> document windows -> manifest
       -> model -> train -> validate -> checkpoint
       -> evaluate -> generate
```

## 模块边界

- `config.py`：typed dataclass 配置、YAML、`--set` 和跨字段校验。
- `data/`：来源读取、清洗、切分、manifest、memmap dataset 和检查。
- `tokenization/`：char 与 byte-level BPE，共享 encode/decode/fingerprint 契约。
- `models/`：bigram、attention、Transformer、registry 和 `ModelOutput`。
- `training/`：optimizer、scheduler、确定性 sampler、trainer 和 checkpoint。
- `evaluation/`：token-weighted loss、perplexity、accuracy、BPC/BPB 和 run comparison。
- `generation/`：greedy、temperature、top-k、top-p、EOS 和 KV cache。
- `inference/`：从 checkpoint 重建模型和内嵌 tokenizer。
- `cli.py`：只负责模块编排，不解析语料或保存可执行 Python 对象。

## 公开契约

模型输入为 `[B,T]`，输出为 `[B,T,V]` logits。manifest identity 绑定源文件、规范化、split、tokenizer、context 和 window policy。checkpoint schema v2 保存模型、优化器、scheduler、scaler、sampler 和 RNG 状态。

`best.pt` 由 valid loss 选择，`last.pt` 用于恢复。不同 manifest 或 tokenizer 的运行不能按 token-level perplexity 排名，比较 CLI 会显式报告这一点。

## 后续路线

后续扩展应先补测试和独立配置，再进行 recorded run：批量生成、更多数据集、Flash Attention、长上下文和多卡训练都不属于当前契约。
