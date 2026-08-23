# 模型

项目包含两个模型族：

| 模型 | 用途 |
| --- | --- |
| `bigram` | 最小 next-token 基线 |
| `transformer_decoder` | pre-LN decoder-only Transformer |

Transformer 由 token embedding、位置编码、causal multi-head attention、残差、GELU MLP、最终归一化和输出投影组成。`readable` 是用于对照的参考实现，`sdpa` 用于 PyTorch attention 实现对比。

建议阅读顺序：`models/bigram.py`、`models/attention.py`、`models/transformer.py`、`models/spec.py`。两种 attention 必须满足相同的 logits 和 cache 契约。
