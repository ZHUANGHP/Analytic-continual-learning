# -*- coding: utf-8 -*-
from typing import Tuple
import torch
from .DatasetWrapper import DatasetWrapper
from torchvision.datasets import ImageNet
from torchvision.transforms import v2 as transforms
from os import path


class ImageNet_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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
        self.dataset = ImageNet(root, split="train" if train else "val")
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
