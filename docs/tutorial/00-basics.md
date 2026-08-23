# Basics

A causal language model predicts the next token at every position. For a sequence `[BOS, x1, x2, x3]`, the training pair is:

```text
input_ids = [BOS, x1, x2]
targets   = [x1,  x2, x3]
```

The model returns logits with shape `[batch, time, vocab]`. Cross-entropy compares each position with its shifted target; padding targets are ignored.

Start with the smallest executable example:

```bash
uv run python examples/01_bigram_forward.py
```

The bigram model maps each input token directly to next-token logits. It has no attention and no long-range interaction, which makes it a useful baseline before reading the Transformer implementation.

Then inspect the causal constraint:

```bash
uv run python examples/02_causal_attention.py
```

Changing a future token must not change logits at earlier positions. This is the central correctness property of decoder-only language modeling.
