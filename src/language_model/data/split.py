"""Deterministic document-level train/validation/test splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from .sources import Document

_SPLIT_NAMES = ("train", "valid", "test")


def _units(documents: list[Document], group_by_source: bool) -> list[list[Document]]:
    if not group_by_source:
        return [[document] for document in documents]
    grouped: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        grouped[document.source].append(document)
    return list(grouped.values())


def _split_by_count(units: list[list[Document]], ratios: list[float]) -> dict[str, list[Document]]:
    count = len(units)
    train_count = max(1, int(count * ratios[0]))
    valid_count = max(1, int(count * ratios[1]))
    if train_count + valid_count >= count:
        train_count = max(1, count - 2)
        valid_count = 1
    unit_splits = {
        "train": units[:train_count],
        "valid": units[train_count : train_count + valid_count],
        "test": units[train_count + valid_count :],
    }
    return {name: [document for unit in selected for document in unit] for name, selected in unit_splits.items()}


def _split_by_characters(units: list[list[Document]], ratios: list[float]) -> dict[str, list[Document]]:
    weighted = [(unit, sum(len(document.text) for document in unit)) for unit in units]
    weighted.sort(key=lambda item: item[1], reverse=True)
    selected: dict[str, list[list[Document]]] = {name: [] for name in _SPLIT_NAMES}
    totals = dict.fromkeys(_SPLIT_NAMES, 0)
    target_total = sum(weight for _, weight in weighted)

    for name, (unit, weight) in zip(_SPLIT_NAMES, weighted[:3], strict=True):
        selected[name].append(unit)
        totals[name] += weight
    for unit, weight in weighted[3:]:
        name = min(
            _SPLIT_NAMES,
            key=lambda candidate: totals[candidate] / max(1.0, target_total * ratios[_SPLIT_NAMES.index(candidate)]),
        )
        selected[name].append(unit)
        totals[name] += weight
    return {name: [document for unit in selected[name] for document in unit] for name in _SPLIT_NAMES}


def split_documents(
    documents: list[Document],
    ratios: list[float],
    seed: int,
    *,
    strategy: str = "document_count",
    group_by_source: bool = False,
) -> dict[str, list[Document]]:
    units = _units(documents, group_by_source)
    if len(units) < 3:
        label = "source groups" if group_by_source else "documents"
        raise ValueError(f"at least three {label} are required")
    random.Random(seed).shuffle(units)
    if strategy == "document_count":
        result = _split_by_count(units, ratios)
    elif strategy == "character_balanced":
        result = _split_by_characters(units, ratios)
    else:
        raise ValueError(f"unknown split strategy: {strategy}")

    identities = [{item.document_id for item in split} for split in result.values()]
    if identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2]:
        raise RuntimeError("document leakage detected while splitting corpus")
    if any(not split for split in result.values()):
        raise RuntimeError("all data splits must be non-empty")
    return result
