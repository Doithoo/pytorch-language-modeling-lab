# Metrics

Evaluation reports token-weighted cross-entropy, perplexity, top-1 accuracy, top-5 accuracy, valid token count, and batch count.

```text
perplexity = exp(mean token cross-entropy)
```

BPC and BPB convert the accumulated negative log likelihood into bits and normalize by source characters or UTF-8 bytes. They are useful context when tokenizers differ; token-level perplexity is not directly comparable across different vocabularies.

Training metrics include update step, train and validation loss, validation perplexity and accuracy, learning rate, processed tokens, elapsed time, token throughput, and CUDA peak memory.

Test metrics are computed after validation selects a final checkpoint. Fixed prompt output is qualitative evidence and does not replace loss-based evaluation.
