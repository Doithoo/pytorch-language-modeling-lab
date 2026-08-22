# 英文教程索引

This project is intentionally taught as a sequence of small contracts:

1. [Environment and data](../guides/first-run.zh-CN.md): install the package, download the pinned corpus, prepare a relocatable manifest, and inspect it.
2. [Bigram baseline](../../examples/01_bigram_forward.py): connect token ids, logits, shifted targets, cross-entropy, and greedy sampling.
3. [Causal attention](../../examples/02_causal_attention.py): observe why future positions must not change past logits.
4. **Decoder-only Transformer**: read embeddings, pre-LN blocks, attention, MLP, residuals, and tied output weights in `src/language_model/models/`.
5. **Training and recovery**: follow `Trainer`, `ResumableBatchSampler`, checkpoint v2, token-weighted accumulation, and the interrupted-vs-continuous integration test.
6. **Evaluation and generation**: compare loss, perplexity, token accuracy, bits-per-character, prompt suites, sampling policies, and KV cache.
7. **Experiment comparison**: prepare separate manifests for character and BPE tokenizers, train with separate run names, and use `lm compare-runs`.

The Chinese guides contain the command-by-command walkthrough today. English mirrors are added as behavior changes land so examples remain executable instead of becoming stale translations.
