# Model Card: Reference Language Models

## Summary

The repository provides small, reproducible causal language models: a character-level bigram baseline and a decoder-only Transformer with learned or rotary positions. The reference checkpoints are trained on Tiny Shakespeare and are intended for learning, regression tests, and controlled comparisons.

## Intended Use

- The project covers tokenization, shifted targets, causal attention, checkpoint recovery, and sampling.
- Compare readable attention, PyTorch SDPA, learned positions, RoPE, and KV-cache implementations.
- Run small offline CPU smoke tests or complete small GPU experiments on Kaggle.

## Out of Scope

These models are not production chatbots, instruction-tuned assistants, factual knowledge systems, safety classifiers, or general-purpose language models. They should not be used for high-stakes decisions, automated publishing, or claims about broad model quality.

## Training Data and Procedure

The published study uses the pinned Tiny Shakespeare source described in the [dataset card](dataset-card.md). Documents are normalized and split before fitting the tokenizer. The model is trained with next-token cross-entropy; validation loss selects `best.pt`, and test metrics are computed after selection.

The canonical study compares readable attention, SDPA, learned positions, RoPE, and shared RoPE cache under small fixed budgets. Results are documented in the [recorded-run index](../recorded-run/README.md).

## Metrics and Limitations

Reported perplexity is token-level and depends on the tokenizer. BPC/BPB provide additional context when tokenizers differ. The corpus is small, English-only, and literary; results do not establish performance on modern web text, multilingual data, long-context tasks, instruction following, factuality, or safety.

Generated text is qualitative reference only. Sampling output can be incoherent, biased by the training corpus, or factually meaningless.

## Reproducibility

Every published study records a source snapshot, manifest identity, tokenizer fingerprint, environment, configuration, metrics, and checkpoint hashes. Checkpoints are intentionally not committed to Git and must be downloaded from the linked Kaggle run and verified against provenance metadata.

## License

Project code is MIT licensed. Check the upstream corpus terms before redistributing data or derived artifacts.
