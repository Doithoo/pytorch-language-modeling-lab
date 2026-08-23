# Models

The tutorial is organized around two model families:

| Model | Purpose |
| --- | --- |
| `bigram` | Minimal next-token baseline |
| `transformer_decoder` | Pre-LN decoder-only Transformer |

The Transformer consists of token embeddings, optional learned positions or RoPE, causal multi-head self-attention, residual connections, GELU MLPs, final normalization, and a tied output projection.

Read the implementation in this order:

1. `src/language_model/models/bigram.py`
2. `src/language_model/models/attention.py`
3. `src/language_model/models/transformer.py`
4. `src/language_model/models/spec.py`

`readable` attention is the reference implementation used for comparisons. `sdpa` delegates attention computation to PyTorch. Both must satisfy the same logits and cache contract, which is covered by model tests.
