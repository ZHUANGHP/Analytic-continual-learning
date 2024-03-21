# -*- coding: utf-8 -*-
"""
Basic analytic linear modules for the analytic learning based CIL [1, 2, 3].

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[3] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. 2024.
"""

import torch
from typing import Optional, Union
from abc import abstractmethod, ABCMeta


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        if self.bias:
            X = torch.concat((X, torch.ones(X.shape[0], 1)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability!"
            "A possible solution is to set self.dtype=torch.double."
        )


class RecursiveLinear(AnalyticLinear):

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        R = torch.eye(in_features, **factory_kwargs) / self.gamma
        self.register_buffer("R", R)
        self.R: torch.Tensor

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """The core code of the ACIL.
        This implementation is different but equivalent to the equations shown in [1],
        which supports mini-batch learning and the gereral CIL setting (e.g. the Si-Blurry).
        The peper for the general CIL settings will be published soon.
        """
        X, Y = X.to(self.weight), Y.to(self.weight)
        if self.bias:
            X = torch.concat((X, torch.ones(X.shape[0], 1)), dim=-1)

        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.concat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.concat((Y, tail), dim=1)

        # ACIL
        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        # Equation (10) of ACIL
        self.R -= self.R @ X.T @ K @ X @ self.R
        # Equation (9) of ACIL
        self.weight += self.R @ X.T @ (Y - X @ self.weight)
