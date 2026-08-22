"""Document-level deterministic train/validation/test splitting."""

from __future__ import annotations

import random

from .sources import Document


def split_documents(documents: list[Document], ratios: list[float], seed: int) -> dict[str, list[Document]]:
    if len(documents) < 3:
        raise ValueError("at least three documents are required")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = max(1, int(count * ratios[0]))
    valid_count = max(1, int(count * ratios[1]))
    if train_count + valid_count >= count:
        train_count = max(1, count - 2)
        valid_count = 1
    result = {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }
    identities = [{item.document_id for item in split} for split in result.values()]
    if identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2]:
        raise RuntimeError("document leakage detected while splitting corpus")
    if any(not split for split in result.values()):
        raise RuntimeError("all data splits must be non-empty")
    return result
