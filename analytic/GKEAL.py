# -*- coding: utf-8 -*-
"""
Implementation of the GKEAL [1].

The GKEAL is a CIL method specially proposed for the few-shot CIL.
But the implementation here is just a simplified version for common CIL settings.
Compared with the method proposed in the paper, we do not perform image augmentation here.
Each sample will only be learned once by default.

References:
[1] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
"""

import torch
from tqdm import tqdm
from typing import Dict, Any, Sequence, Optional
from torch._prims_common import DeviceLikeType
from .Learner import loader_t
from .ACIL import ACIL, ACILLearner
from .Buffer import GaussianKernel
from .AnalyticLinear import AnalyticLinear, RecursiveLinear


class GKEAL(ACIL):
    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 512,
        gamma: float = 1e-3,
        sigma: float = 10,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ):
        super().__init__(
            backbone_output, backbone, buffer_size, gamma, device, dtype, linear
        )
        self.buffer = GaussianKernel(
            torch.zeros((self.buffer_size, self.backbone_output)),
            sigma,
            device=device,
            dtype=dtype,
        )


class GKEALLearner(ACILLearner):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.initialized = False
        # The width-adjusting parameter β controls the width of the Gaussian kernels.
        # There is a comfortable range for σ at around [5, 15] for CIFAR-100 and ImageNet-1k
        # that gives good results, where β = 1 / (2σ²).
        self.sigma = args["sigma"]
        super().__init__(args, backbone, backbone_output, device, all_devices)

    def make_model(self) -> None:
        self.model = GKEAL(
            self.backbone_output,
            self.backbone,
            self.buffer_size,
            self.gamma,
            self.sigma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )

    @torch.no_grad()
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        desc: str = "Incremental Learning",
    ) -> None:
        torch.cuda.empty_cache()
        if self.initialized:
            return super().learn(data_loader, incremental_size, desc)
        total_X = []
        total_y = []
        for X, y in tqdm(data_loader, desc="Selecting center vectors"):
            X: torch.Tensor = X.to(self.device, non_blocking=True)
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            X = self.backbone(X)
            total_X.append(X)
            total_y.append(y)

        self.model.buffer.init(torch.cat(total_X), self.buffer_size)
        torch.cuda.empty_cache()
        for X, y in tqdm(zip(total_X, total_y), total=len(total_X), desc=desc):
            X = self.model.buffer(X)
            Y = torch.nn.functional.one_hot(y, incremental_size)
            self.model.analytic_linear.fit(X, Y)
        self.model.analytic_linear.update()
        self.initialized = True
