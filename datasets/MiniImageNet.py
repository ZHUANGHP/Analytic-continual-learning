# -*- coding: utf-8 -*-
from typing import Tuple
import torch
from .DatasetWrapper import DatasetWrapper
from torchvision.datasets import ImageFolder  # mini-ImageNet通常作为文件夹形式提供
from torchvision.transforms import v2 as transforms
from os import path


class MiniImageNet_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 100  # mini-ImageNet包含100个类
    mean = (0.47302529,0.44856255,0.40276794)  # 经过脚本计算得到该数值
    std = (0.28168869,0.27359548,0.28692064)  # 同上

    basic_transform = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(0.1),
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
        root = path.expanduser(root)
        self.dataset = ImageFolder(root=root)  # 假设mini-ImageNet作为标准图像文件夹结构提供
        super().__init__(
            [label for _, label in self.dataset.samples],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
