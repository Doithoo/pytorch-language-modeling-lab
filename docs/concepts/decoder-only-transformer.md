# Decoder-only Transformer

The Transformer uses pre-LN decoder blocks:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Each attention layer projects Q, K, and V, applies a causal mask, and returns a residual update. Learned positional embeddings add a trainable vector before the blocks. RoPE instead rotates Q and K at their sequence positions.

The readable implementation is the reference path. SDPA is a controlled implementation alternative. KV cache stores per-layer K/V tensors for incremental generation; when the cache reaches the context limit, generation rebuilds it from the recent context.

The common model contract is `[B, T]` input ids, `[B, T, V]` logits, optional token-weighted loss, and optional past key/value tensors. See the [code tour](code-tour.md) for the reading order.
