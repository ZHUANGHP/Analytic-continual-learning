# -*- coding: utf-8 -*-

from typing import Callable, Iterable, Optional
from torch.utils.data import Dataset, Subset
try:
    from torch.utils.data.dataset import T_co
except ImportError:
    from torch.utils._ordered_set import T_co
from abc import ABCMeta
from random import Random
from numpy import repeat
from itertools import chain


class DatasetWrapper(Dataset[T_co], metaclass=ABCMeta):
    basic_transform: Callable[[T_co], T_co]
    augment_transform: Callable[[T_co], T_co]

    def __init__(
        self,
        labels: Iterable[int],
        base_ratio: float,
        num_phases: int,
        augment: bool,
        inplace_repeat: int = 1,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        # Type hints
        self.dataset: Dataset[T_co]
        self.num_classes: int

        # Initialization
        super().__init__()
        self.inplace_repeat = inplace_repeat
        self.base_ratio = base_ratio
        self.num_phases = num_phases
        self.base_size = int(self.num_classes * self.base_ratio)
        self.incremental_size = self.num_classes - self.base_size
        self.phase_size = self.incremental_size // num_phases if num_phases > 0 else 0
        # Create a list of indices for each class
        self.class_indices: list[list[int]] = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        self._transform = self.augment_transform if augment else self.basic_transform
        # Shuffle the class indices
        self.real_labels: list[int] = list(range(self.num_classes))
        if shuffle_seed is not None:
            Random(shuffle_seed).shuffle(self.real_labels)
            Random(shuffle_seed).shuffle(self.class_indices)

    def __getitem__(self, index: int) -> T_co:
        return self._transform(self.dataset[index])

    def _subset(self, label_begin: int, label_end: int) -> Subset[T_co]:
        sub_ids = tuple(chain.from_iterable(self.class_indices[label_begin:label_end]))
        return Subset(self, repeat(sub_ids, self.inplace_repeat).tolist())

    def subset_at_phase(self, phase: int) -> Subset[T_co]:
        if phase == 0:
            return self._subset(0, self.base_size)
        return self._subset(
            self.base_size + (phase - 1) * self.phase_size,
            self.base_size + phase * self.phase_size,
        )

    def subset_until_phase(self, phase: int) -> Subset[T_co]:
        return self._subset(
            0,
            self.base_size + phase * self.phase_size,
        )
