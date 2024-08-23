# -*- coding: utf-8 -*-
"""
Implementation of the AEF-OCL [1], an analytic method for imbalanced continual learning.

References:
[1] Zhuang, Huiping, et al.
    "Online Analytic Exemplar-Free Continual Learning with Large Models for Imbalanced Autonomous Driving Task"
    arXiv preprint arXiv:2405.17779 (2024).
"""

from copy import deepcopy
import torch
from tqdm import tqdm
from .ACIL import ACILLearner, ACIL
from .AnalyticLinear import AnalyticLinear, RecursiveLinear

__all__ = ["AEFOCL", "AEFOCLLearner"]


class AEFOCL(ACIL):
    """
    Network structure of the AEF-OCL [1], an analytic method for imbalanced continual learning.

    References:
    [1] Zhuang, Huiping, et al.
        "Online Analytic Exemplar-Free Continual Learning with Large Models for Imbalanced Autonomous Driving Task"
        arXiv preprint arXiv:2405.17779 (2024).
    """

    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 8192,
        gamma: float = 1e-3,
        noise: float = 1,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__(
            backbone_output, backbone, buffer_size, gamma, device, dtype, linear
        )
        self._linear_log = dict()
        # History prototype
        self.noise = noise
        # Expectation of the prototypes E[X]
        self.register_buffer("ex", torch.zeros((0, backbone_output), dtype=torch.double))
        self.ex: torch.Tensor
        # Expectation of the squares of the prototypes E[X^2]
        self.register_buffer("ex2", torch.zeros((0, backbone_output), dtype=torch.double))
        self.ex2: torch.Tensor
        # Number of the samples of the prototypes
        self.register_buffer("cnt", torch.zeros((0,), dtype=torch.long))
        self.cnt: torch.Tensor
        # Set the device
        self.to(device)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        for name, buffer in self._linear_log.items():
            self.analytic_linear.register_buffer(name, buffer)
        self._linear_log.clear()

        X = self.backbone(X)
        if (increment_size := int(y.max().item()) - self.ex.shape[0] + 1) > 0:
            # self.cnt
            tail = torch.zeros((increment_size,)).to(self.cnt)
            self.cnt = torch.concat((self.cnt, tail), dim=0)
            # self.ex
            tail = torch.zeros((increment_size, self.ex.shape[1])).to(self.ex)
            self.ex = torch.concat((self.ex, tail), dim=0)
            # self.ex2
            tail = torch.zeros((increment_size, self.ex2.shape[1])).to(self.ex2)
            self.ex2 = torch.concat((self.ex2, tail), dim=0)

        labels, counts = torch.unique(y, return_counts=True)
        self.cnt[labels] += counts
        for i in labels:
            X_i = X[y == i]
            # Calculate E[X]
            self.ex[i] += torch.sum(X_i.to(self.ex), dim=0)
            # Calculate E[X^2]
            self.ex2[i] += torch.sum(torch.square(X_i.to(self.ex2)), dim=0)

        X = self.buffer(X)
        Y = torch.nn.functional.one_hot(y)
        self.analytic_linear.fit(X, Y)

    def update(self) -> None:
        peak_cnt = int(self.cnt.max())
        mean = self.proto_mean
        std = self.proto_std
        print("Counts:", self.cnt.tolist())

        # Backup the iterative classifier
        for name, buffer in self.analytic_linear.named_buffers():
            self._linear_log[name] = buffer.clone().detach()

        aug_bar = tqdm(
            desc="Augmenting",
            total=(peak_cnt * len(self.cnt.nonzero()) - int(self.cnt.sum())),
        )
        for i in self.cnt.nonzero():
            i = int(i.item())
            rest_cnt = int(peak_cnt - self.cnt[i])
            while rest_cnt > 0:
                fill_cnt = min(rest_cnt, 8192)
                fill_y = torch.empty((fill_cnt,), dtype=torch.long).fill_(i)
                fill_proto = torch.randn((fill_cnt, self.buffer.in_features)).to(
                    self.buffer.weight
                )
                fill_proto = (
                    fill_proto * std[i][None, :] * self.noise + mean[i][None, :]
                )
                fill_proto = self.buffer(fill_proto)
                fill_y = torch.nn.functional.one_hot(fill_y)
                self.analytic_linear.fit(fill_proto, fill_y)
                aug_bar.update(fill_cnt)
                rest_cnt -= fill_cnt
        self.analytic_linear.update()

    @property
    def proto_mean(self) -> torch.Tensor:
        return self.ex / self.cnt[:, None]

    @property
    def proto_std(self) -> torch.Tensor:
        std = self.ex2 / self.cnt[:, None] - torch.square(self.proto_mean)
        std[torch.isnan(std)] = 0
        assert (std >= 0).all()
        proto_std = torch.sqrt(std * (self.cnt / (self.cnt - 1))[:, None])
        proto_std[torch.isnan(proto_std)] = 0
        return proto_std


class AEFOCLLearner(ACILLearner):
    """
    Learner of the AEF-OCL [1], an analytic method for imbalanced continual learning.

    References:
    [1] Zhuang, Huiping, et al.
        "Online Analytic Exemplar-Free Continual Learning with Large Models for Imbalanced Autonomous Driving Task"
        arXiv preprint arXiv:2405.17779 (2024).
    """

    def make_model(self) -> None:
        self.model = AEFOCL(
            self.backbone_output,
            self.backbone,
            self.buffer_size,
            self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )
