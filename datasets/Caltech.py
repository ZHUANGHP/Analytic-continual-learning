# -*- coding: utf-8 -*-
from typing import Tuple
import torch
from .DatasetWrapper import DatasetWrapper
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from os import path


class Caltech256_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 257  # 包含一个背景类
    mean = (0.485, 0.456, 0.406)  # 常用的ImageNet均值和标准差
    std = (0.229, 0.224, 0.225)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(),
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
        self.dataset = ImageFolder(root=path.expanduser(root))  # 直接使用提供的root路径
        super().__init__(
            [label for _, label in self.dataset.samples],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )



class Caltech101_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 102
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(),
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
        self.dataset = ImageFolder(root=path.expanduser(root))  # 直接使用提供的root路径
        super().__init__(
            [label for _, label in self.dataset.samples],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )