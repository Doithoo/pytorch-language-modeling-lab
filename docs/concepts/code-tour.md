# Code Tour

Read the package in this order:

1. `tokenization/char.py`: text and ids.
2. `data/dataset.py`: shifted windows and padding.
3. `models/bigram.py`: the smallest logits contract.
4. `models/attention.py`: QKV, masks, and cache.
5. `models/transformer.py`: decoder blocks and loss.
6. `evaluation/metrics.py`: token-weighted metrics.
7. `training/trainer.py`: accumulation, validation, and saves.
8. `training/checkpoint.py`: serializable recovery state.
9. `generation/`: filtering and incremental decoding.
10. `training/train.py` and `cli.py`: orchestration.

The CLI composes these modules. It does not parse corpora, construct attention, or serialize executable model objects.
