# 模型与配置选择

| 配置/模型 | 用途 | 建议环境 |
| --- | --- | --- |
| `bigram_tiny.yaml` | Token transition、logits、loss 和采样 | 本地 CPU |
| `char_tiny.yaml` | Causal attention 与 Transformer | 本地 dry run / Kaggle |
| `bpe_tiny.yaml` | 子词词表与序列长度权衡 | Kaggle |
| `kaggle_char_small.yaml` | 一次完整的 readable attention 训练 | Kaggle GPU |

`bigram` 每个输入 token 直接索引下一 token logits，不使用上下文交互。它适合作为最小 baseline，但无法建模长依赖。

`transformer_decoder` 使用 learned 或 rotary positions、pre-LN blocks、multi-head causal attention、GELU MLP 和 tied embeddings。`attention_implementation` 可选 readable 或 SDPA。默认 small 配置是 6 层、384 hidden、6 heads、256 context；实际参数量在运行开始时记录。

扩展模型时应保持 `ModelOutput`、`[B,T,V]` logits、padding loss 与 cache/no-cache 契约，并在 registry 中显式注册。不要在 checkpoint 中保存任意模型对象。
