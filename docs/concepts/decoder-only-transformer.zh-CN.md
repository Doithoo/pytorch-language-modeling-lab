# Decoder-only Transformer

输入 token 先经过 token embedding。`learned` 模式再加可训练 position embedding；`rotary` 模式在每层 attention 中旋转 Q/K。每个 pre-LN decoder block 执行：

```text
x = x + CausalSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

attention 把隐藏维度拆成多个 head，计算：

```text
softmax(QK^T / sqrt(head_dim) + causal_mask) V
```

causal mask 只允许 query 位置看到相同位置和更早的 key。测试会修改未来 token，并断言过去位置的 logits 不变。

MLP 使用 `Linear -> GELU -> Linear -> Dropout`。最后经过 LayerNorm 和 lm head 得到 `[B,T,V]`。默认把 lm head 权重与 token embedding 绑定，减少参数并共享输入/输出 token 表示。

RoPE cos/sin 按最大 context length 在模型级预计算，每个 forward 按当前 positions 和 autocast dtype 取一次切片，并由全部 decoder block 共享。table 使用非持久 buffer，不进入 checkpoint state dict，因此动态 RoPE checkpoint 与 `0.2.1` cache 实现兼容。性能试验见 [RoPE cache 记录](../recorded-run/v2/rope-cache/README.zh-CN.md)。

生成时第一次 forward 计算 prompt 的 key/value；后续每一步只输入新 token，并复用每层 KV cache。缓存达到 context length 后，生成器截取最近上下文并重建缓存。测试比较 cache 与完整 forward 的最后位置 logits。

这是可读参考实现，不追求 fused kernel 的极限吞吐。性能优化必须先与该路径比较数值和生成结果。
