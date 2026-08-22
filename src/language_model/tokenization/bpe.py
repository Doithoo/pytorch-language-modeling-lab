"""BPE tokenizer backed by Hugging Face tokenizers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ..utils import write_json
from .base import LanguageModelTokenizer, TokenizerError


class BPETokenizer(LanguageModelTokenizer):
    def __init__(self, tokenizer: Tokenizer, special_tokens: list[str]) -> None:
        if len(special_tokens) != 4 or len(set(special_tokens)) != 4:
            raise TokenizerError("expected four unique special tokens: pad, bos, eos, unk")
        self._tokenizer = tokenizer
        self.special_tokens = list(special_tokens)
        missing = [token for token in special_tokens if tokenizer.token_to_id(token) is None]
        if missing:
            raise TokenizerError(f"BPE tokenizer is missing special tokens: {missing}")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        special_tokens: list[str],
        vocab_size: int,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        if vocab_size < 32:
            raise TokenizerError("BPE vocab_size must be at least 32")
        tokenizer = Tokenizer(models.BPE(unk_token=special_tokens[3]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer, special_tokens)

    def _special_id(self, index: int) -> int:
        token_id = self._tokenizer.token_to_id(self.special_tokens[index])
        if token_id is None:
            raise TokenizerError(f"special token missing: {self.special_tokens[index]}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._special_id(0)

    @property
    def bos_id(self) -> int:
        return self._special_id(1)

    @property
    def eos_id(self) -> int:
        return self._special_id(2)

    @property
    def unk_id(self) -> int:
        return self._special_id(3)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        if any(index < 0 or index >= self.vocab_size for index in ids):
            raise TokenizerError("token id out of range")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "type": "bpe",
            "special_tokens": self.special_tokens,
            "tokenizer_json": self._tokenizer.to_str(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> BPETokenizer:
        if payload.get("schema_version") != 1 or payload.get("type") != "bpe":
            raise TokenizerError("unsupported BPE tokenizer payload")
        return cls(Tokenizer.from_str(str(payload["tokenizer_json"])), list(payload["special_tokens"]))

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text(self._tokenizer.to_str(pretty=True), encoding="utf-8")
        write_json(
            target / "metadata.json",
            {
                "type": "bpe",
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "fingerprint": self.fingerprint(),
            },
        )

    @classmethod
    def load_directory(cls, directory: str | Path) -> BPETokenizer:
        target = Path(directory)
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        return cls(Tokenizer.from_file(str(target / "tokenizer.json")), list(metadata["special_tokens"]))
