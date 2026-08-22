"""Memory-mapped, document-aware next-token window dataset."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import sha256_file
from .manifest import ManifestError, load_manifest


class TokenWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        processed_root: str | Path | None = None,
        verify_hash: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.context_length = int(self.manifest["context_length"])
        self.stride = int(self.manifest["stride"])
        self.pad_id = int(self.manifest["tokenizer"]["special_token_ids"]["pad"])
        split_metadata = self.manifest["splits"][split]
        if processed_root is not None:
            root = Path(processed_root)
        else:
            root = Path(self.manifest["processed_root"])
            if not root.is_absolute():
                root = self.manifest_path.parent / root
        self.data_path = root / split_metadata["file"]
        if not self.data_path.is_file():
            raise FileNotFoundError(f"prepared token file not found: {self.data_path}")
        if verify_hash and sha256_file(self.data_path) != split_metadata["sha256"]:
            raise ManifestError(f"prepared token hash mismatch for split {split}")
        self.tokens = np.memmap(self.data_path, dtype=np.uint32, mode="r")
        self.offsets: list[dict[str, Any]] = list(split_metadata["offsets"])
        self.cumulative: list[int] = []
        total = 0
        for item in self.offsets:
            total += int(item["windows"])
            self.cumulative.append(total)
        if total != int(split_metadata["windows"]):
            raise ManifestError(f"window count mismatch for split {split}")
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        document_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[document_index - 1] if document_index else 0
        local_window = index - previous
        document = self.offsets[document_index]
        start = int(document["start"]) + local_window * self.stride
        end = min(start + self.context_length + 1, int(document["end"]))
        sequence = np.asarray(self.tokens[start:end], dtype=np.int64)
        padded = np.full(self.context_length + 1, self.pad_id, dtype=np.int64)
        padded[: len(sequence)] = sequence
        input_ids = torch.from_numpy(padded[:-1].copy())
        targets = torch.from_numpy(padded[1:].copy())
        return input_ids, targets
