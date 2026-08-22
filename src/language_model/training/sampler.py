"""Deterministic batch ordering that can resume from a consumed micro-step."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

SAMPLER_ALGORITHM = "epoch-seeded-random-v1"


class ResumableBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> None:
        if dataset_size < 1 or batch_size < 1:
            raise ValueError("dataset_size and batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.start_micro_step = 0

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def set_start_micro_step(self, micro_step: int) -> None:
        if micro_step < 0:
            raise ValueError("micro_step must be non-negative")
        self.start_micro_step = micro_step

    def _indices(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(self.dataset_size))
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        epoch, first_batch = divmod(self.start_micro_step, self.batches_per_epoch)
        indices = self._indices(epoch)
        for batch_index in range(first_batch, self.batches_per_epoch):
            start = batch_index * self.batch_size
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
        self.start_micro_step = (epoch + 1) * self.batches_per_epoch

    def __len__(self) -> int:
        _, first_batch = divmod(self.start_micro_step, self.batches_per_epoch)
        return self.batches_per_epoch - first_batch

    def contract(self) -> dict[str, int | str | bool]:
        return {
            "algorithm": SAMPLER_ALGORITHM,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "batches_per_epoch": self.batches_per_epoch,
        }
