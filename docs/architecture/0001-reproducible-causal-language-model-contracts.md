# Reproducible Causal Language Model Contracts

This architecture decision is accepted and verified by the current contract tests.

The project treats source data, split identity, tokenizer payload, model configuration, training state, generation parameters, and runtime environment as separate pieces of experiment state. Manifests and checkpoint metadata bind those pieces together.

The common model contract is next-token prediction with `[B, T, V]` logits. `best.pt` is selected from validation loss; test is evaluated after model selection. Checkpoints contain state dictionaries and serializable data rather than Python model objects.

The project does not promise cross-hardware bitwise generation, distributed training, production serving, or factual and safe output. See the [Chinese ADR](0001-reproducible-causal-language-model-contracts.zh-CN.md) for the full decision record.
