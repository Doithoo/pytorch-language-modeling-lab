# 基础

因果语言模型在每个位置预测下一个 token：

```text
input_ids = [BOS, x1, x2]
targets   = [x1,  x2, x3]
```

模型输出 `[batch, time, vocab]` 的 logits，交叉熵只统计非 padding target。

先运行最小 bigram 示例：

```bash
uv run python examples/01_bigram_forward.py
uv run python examples/02_causal_attention.py
```

第二个示例验证未来 token 的改变不会影响更早位置的 logits，这是 decoder-only 模型的核心正确性约束。
