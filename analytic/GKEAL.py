# -*- coding: utf-8 -*-
"""
Implementation of the GKEAL [1].

References:
[1] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
"""

from tqdm import tqdm
from .Learner import loader_t
from .ACIL import ACIL, ACILLearner
import torch
from .Buffer import GaussianKernel
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
from typing import Dict, Any


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
    ) -> None:
        self.initialized = False
        self.sigma = args["sigma"]
        super().__init__(args, backbone, backbone_output, device)

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
        for X, y in tqdm(data_loader, desc=desc):
            X: torch.Tensor = X.to(self.device, non_blocking=True)
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            X = self.backbone(X)
            total_X.append(X)
            total_y.append(y)
        X = torch.concat(total_X)
        y = torch.concat(total_y)

        BATCH_SIZE = int(total_y[0].shape[0])
        total_X.clear()
        total_y.clear()
        torch.cuda.empty_cache()
        self.model.buffer.init(X, self.buffer_size)
        for i in tqdm(range(0, X.shape[0], BATCH_SIZE), desc=desc):
            end = min(i + BATCH_SIZE, X.shape[0])
            X_batch = X[i : end]
            X_batch = self.model.buffer(X_batch)
            Y_batch = torch.nn.functional.one_hot(y[i : end], incremental_size)
            self.model.analytic_linear.fit(X_batch, Y_batch)
        self.model.analytic_linear.update()
        self.initialized = True
