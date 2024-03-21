# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import v2 as transforms
from .DatasetWrapper import DatasetWrapper


class MNIST_(DatasetWrapper[tuple[Tensor, int]]):
    num_classes = 10
    mean = (0.13066047627384287048,)
    std = (0.30524474224261827502,)

    basic_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
        ]
    )
    augment_transform = basic_transform

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
        self.dataset = MNIST(root, train=train, download=True)
        super().__init__(
            self.dataset.targets.tolist(),
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
