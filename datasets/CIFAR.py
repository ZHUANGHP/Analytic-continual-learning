# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import v2 as transforms
from typing import Tuple
from .DatasetWrapper import DatasetWrapper


class CIFAR10_(DatasetWrapper[Tuple[Tensor, int]]):
    num_classes = 10
    mean = (0.49139967861519607843, 0.48215840839460784314, 0.44653091444546568627)
    std = (0.21117028181572183225, 0.20857934290628859220, 0.21205155387102001073)
    basic_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

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
        self.dataset = CIFAR10(root, train=train, download=True)
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )


class CIFAR100_(DatasetWrapper[Tuple[Tensor, int]]):
    num_classes = 100
    mean = (0.50707515923713235294, 0.48654887331495098039, 0.44091784336703431373)
    std = (0.26733428848992695514, 0.25643846542136995765, 0.27615047402246589731)
    # std = (0.21103932286924015314, 0.20837755491382136483, 0.21551368222930648019)
    basic_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

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
        self.dataset = CIFAR100(root, train=train, download=True)
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )


if __name__ == "__main__":
    dataset_train = CIFAR100_(
        "~/.dataset", train=True, base_ratio=0.1, num_phases=3, augment=True
    )
    dataset_test = CIFAR100_(
        "~/.dataset", train=False, base_ratio=0.1, num_phases=3, augment=False
    )

    for X, y in dataset_train.subset_at_phase(0):
        assert X.shape == (3, 32, 32)
    for X, y in dataset_test.subset_at_phase(0):
        assert X.shape == (3, 32, 32)
    print("test passed")
