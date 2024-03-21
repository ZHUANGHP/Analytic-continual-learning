# -*- coding: utf-8 -*-

from .DatasetWrapper import DatasetWrapper
from .MNIST import MNIST_ as MNIST
from .CIFAR import CIFAR10_ as CIFAR10
from .CIFAR import CIFAR100_ as CIFAR100
from .ImageNet import ImageNet_ as ImageNet
from typing import Union
from .Features import Features


__all__ = [
    "load_dataset",
    "dataset_list",
    "MNIST",
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
    "DatasetWrapper",
    "Features",
]

dataset_list = {
    "MNIST": MNIST,
    "CIFAR-10": CIFAR10,
    "CIFAR-100": CIFAR100,
    "ImageNet-1k": ImageNet,
}


def load_dataset(
    name: str,
    root: str,
    train: bool,
    base_ratio: float,
    num_phases: int,
    augment: bool = False,
    inplace_repeat: int = 1,
    shuffle_seed: int | None = None,
    *args,
    **kwargs
) -> Union[MNIST, CIFAR10, CIFAR100, ImageNet]:
    return dataset_list[name](
        root=root,
        train=train,
        base_ratio=base_ratio,
        num_phases=num_phases,
        augment=augment,
        inplace_repeat=inplace_repeat,
        shuffle_seed=shuffle_seed,
        *args,
        **kwargs
    )
