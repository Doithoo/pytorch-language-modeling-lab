"""Deterministic character tokenizer used by the first learning path."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..utils import write_json
from .base import LanguageModelTokenizer, TokenizerError


class CharTokenizer(LanguageModelTokenizer):
    def __init__(self, vocabulary: list[str], special_tokens: list[str]) -> None:
        if len(special_tokens) != 4 or len(set(special_tokens)) != 4:
            raise TokenizerError("expected four unique special tokens: pad, bos, eos, unk")
        if len(vocabulary) != len(set(vocabulary)):
            raise TokenizerError("character vocabulary contains duplicates")
        if any(token in vocabulary for token in special_tokens):
            raise TokenizerError("special tokens must not also appear as vocabulary entries")
        self.special_tokens = list(special_tokens)
        self.vocabulary = list(vocabulary)
        self._tokens = self.special_tokens + self.vocabulary
        self._token_to_id = {token: index for index, token in enumerate(self._tokens)}

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        special_tokens: list[str],
        min_frequency: int = 1,
    ) -> CharTokenizer:
        counts: Counter[str] = Counter()
        seen = False
        for text in texts:
            seen = True
            counts.update(text)
        if not seen or not counts:
            raise TokenizerError("cannot train a character tokenizer on empty text")
        vocabulary = sorted(character for character, count in counts.items() if count >= min_frequency)
        if not vocabulary:
            raise TokenizerError("min_frequency removed every character")
        return cls(vocabulary, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self._token_to_id.get(character, self.unk_id) for character in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        for index in ids:
            if index < 0 or index >= self.vocab_size:
                raise TokenizerError(f"token id out of range: {index}")
            if index < len(self.special_tokens):
                if not skip_special_tokens:
                    pieces.append(self._tokens[index])
            else:
                pieces.append(self._tokens[index])
        return "".join(pieces)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "type": "char",
            "special_tokens": self.special_tokens,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CharTokenizer:
        if payload.get("schema_version") != 1 or payload.get("type") != "char":
            raise TokenizerError("unsupported character tokenizer payload")
        return cls(list(payload["vocabulary"]), list(payload["special_tokens"]))

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "tokenizer.json", self.to_payload())
        write_json(
            target / "metadata.json",
            {"type": "char", "vocab_size": self.vocab_size, "fingerprint": self.fingerprint()},
        )
