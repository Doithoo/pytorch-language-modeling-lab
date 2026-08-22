"""Tokenizer interface and serialization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..utils import sha256_json


class TokenizerError(ValueError):
    """Raised for invalid tokenizer training, loading, or text input."""


class LanguageModelTokenizer(ABC):
    special_tokens: list[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def pad_id(self) -> int: ...

    @property
    @abstractmethod
    def bos_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @property
    @abstractmethod
    def unk_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...

    @abstractmethod
    def to_payload(self) -> dict[str, Any]: ...

    def fingerprint(self) -> str:
        return sha256_json(self.to_payload())

    @abstractmethod
    def save(self, directory: str | Path) -> None: ...
