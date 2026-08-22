# 示例

按顺序运行：

1. `01_bigram_forward.py`：打印输入、`[B,T,V]` logits 和 next-token loss。
2. `02_causal_attention.py`：修改未来 token，验证过去 logits 不变。

```bash
uv run python examples/01_bigram_forward.py
uv run python examples/02_causal_attention.py
```

示例保持短小并使用合成 tensor。真实数据准备、训练、评估和生成使用 `lm` CLI，避免在示例中复制完整工程流程。
