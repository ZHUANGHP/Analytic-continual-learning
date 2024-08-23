# -*- coding: utf-8 -*-

import torch
from os import path
from .DatasetWrapper import DatasetWrapper
from torch.utils.data import TensorDataset
from torchvision.transforms import v2 as transforms


class Features(DatasetWrapper[tuple[torch.Tensor, torch.LongTensor]]):
    basic_transform = transforms.Identity()
    augment_transform = transforms.Identity()

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        assert augment == False, "Augmentation is not supported for Features dataset"

        if train:
            X: torch.Tensor = torch.load(path.join(root, "X_train.pt"), weights_only=True)
            y: torch.Tensor = torch.load(path.join(root, "y_train.pt"), weights_only=True)
        else:
            X: torch.Tensor = torch.load(path.join(root, "X_test.pt"), weights_only=True)
            y: torch.Tensor = torch.load(path.join(root, "y_test.pt"), weights_only=True)

        y = y.to(torch.long, non_blocking=True)
        self.dataset = TensorDataset(X, y)  # type: ignore
        self.num_classes = int(y.max().item()) + 1

        super().__init__(
            y.numpy().tolist(),
            base_ratio,
            num_phases,
            False,
            inplace_repeat,
            shuffle_seed,
        )
