"""Tokenizer factories and checkpoint payload loading."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import TokenizerConfig
from .base import LanguageModelTokenizer, TokenizerError
from .bpe import BPETokenizer
from .char import CharTokenizer


def train_tokenizer(texts: Iterable[str], cfg: TokenizerConfig) -> LanguageModelTokenizer:
    if cfg.name == "char":
        return CharTokenizer.train(texts, cfg.special_tokens, cfg.min_frequency)
    if cfg.name == "bpe" and cfg.vocab_size is not None:
        return BPETokenizer.train(texts, cfg.special_tokens, cfg.vocab_size, cfg.min_frequency)
    raise TokenizerError(f"unsupported tokenizer configuration: {cfg.name}")


def tokenizer_from_payload(payload: dict[str, Any]) -> LanguageModelTokenizer:
    kind = payload.get("type")
    if kind == "char":
        return CharTokenizer.from_payload(payload)
    if kind == "bpe":
        return BPETokenizer.from_payload(payload)
    raise TokenizerError(f"unsupported tokenizer payload type: {kind!r}")


def load_tokenizer(directory: str | Path) -> LanguageModelTokenizer:
    target = Path(directory)
    metadata_path = target / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"tokenizer metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tokenizer: LanguageModelTokenizer
    if metadata.get("type") == "char":
        payload = json.loads((target / "tokenizer.json").read_text(encoding="utf-8"))
        tokenizer = CharTokenizer.from_payload(payload)
    elif metadata.get("type") == "bpe":
        tokenizer = BPETokenizer.load_directory(target)
    else:
        raise TokenizerError(f"unknown tokenizer type: {metadata.get('type')!r}")
    if tokenizer.fingerprint() != metadata.get("fingerprint"):
        raise TokenizerError("tokenizer fingerprint does not match metadata")
    return tokenizer


__all__ = [
    "BPETokenizer",
    "CharTokenizer",
    "LanguageModelTokenizer",
    "TokenizerError",
    "load_tokenizer",
    "tokenizer_from_payload",
    "train_tokenizer",
]
