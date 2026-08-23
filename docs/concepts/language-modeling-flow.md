# Language Modeling Flow

A causal language model factorizes a sequence as:

```text
P(x1, ..., xn) = product of P(xt | x1, ..., x(t-1))
```

This repository represents each document with BOS/EOS markers, creates shifted input and target windows, and ignores padded target positions in cross-entropy. Evaluation accumulates loss by valid target token rather than averaging batches equally.

The data identity is established before tokenization and window creation. The tokenizer is fitted only on train documents, while valid and test documents use the frozen tokenizer. This prevents information from later splits entering the model vocabulary.

Training uses train for optimization, valid for selecting `best.pt`, and test once after the model decision is frozen. Generation samples are supplementary qualitative evidence.
