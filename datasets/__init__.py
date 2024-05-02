# -*- coding: utf-8 -*-

from .DatasetWrapper import DatasetWrapper
from .MNIST import MNIST_ as MNIST
from .CIFAR import CIFAR10_ as CIFAR10
from .CIFAR import CIFAR100_ as CIFAR100
from .ImageNet import ImageNet_ as ImageNet
from .MiniImageNet import MiniImageNet_ as MiniImageNet  # 导入MiniImageNet
from .Caltech import Caltech256_ as Caltech256  # 导入Caltech256
from .Caltech import Caltech101_ as Caltech101  # 导入Caltech101
from typing import Union
from .Features import Features

__all__ = [
    "load_dataset",
    "dataset_list",
    "MNIST",
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
    "MiniImageNet",  # 添加MiniImageNet到导出列表
    "Caltech256",  # 添加Caltech256到导出列表
    "Caltech101",  # 添加Caltech101到导出列表
    "DatasetWrapper",
    "Features",
]

dataset_list = {
    "MNIST": MNIST,
    "CIFAR-10": CIFAR10,
    "CIFAR-100": CIFAR100,
    "ImageNet-1k": ImageNet,
    "MiniImageNet": MiniImageNet,  # 添加MiniImageNet到数据集列表
    "Caltech-256": Caltech256,  # 添加Caltech256到数据集列表
    "Caltech-101": Caltech101,  # 添加Caltech101到数据集列表
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
) -> Union[MNIST, CIFAR10, CIFAR100, ImageNet, MiniImageNet, Caltech256, Caltech101]:  # 添加MiniImageNet, Caltech256和Caltech101到返回类型
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
