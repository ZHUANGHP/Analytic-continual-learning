# -*- coding: utf-8 -*-
"""
Implementation of the DS-AL [1].

References:
[1] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
"""

import torch
from .ACIL import ACILLearner
from typing import Callable, Dict, Any, Optional, Sequence
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
from .Buffer import activation_t, RandomBuffer
from torch._prims_common import DeviceLikeType


class DSAL(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        backbone: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Flatten(),
        expansion_size: int = 8192,
        gamma_main: float = 1e-3,
        gamma_comp: float = 1e-3,
        C: float = 1,
        activation_main: activation_t = torch.relu,
        activation_comp: activation_t = torch.tanh,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.expansion_size = expansion_size
        self.buffer = RandomBuffer(
            backbone_output,
            expansion_size,
            activation=torch.nn.Identity(),
            **factory_kwargs
        )
        # The main stream
        self.activation_main = activation_main
        self.main_stream = linear(expansion_size, gamma_main, **factory_kwargs)
        # The compensation stream
        self.C = C
        self.activation_comp = activation_comp
        self.comp_stream = linear(expansion_size, gamma_comp, **factory_kwargs)
        self.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.buffer(self.backbone(X))
        X_main = self.main_stream(self.activation_main(X))
        X_comp = self.comp_stream(self.activation_comp(X))
        return X_main + self.C * X_comp

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, increase_size: int) -> None:
        num_classes = max(self.main_stream.out_features, int(y.max().item()) + 1)
        Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)
        X = self.buffer(self.backbone(X))

        # Train the main stream
        X_main = self.activation_main(X)
        self.main_stream.fit(X_main, Y_main)
        self.main_stream.update()

        # Previous label cleansing (PLC)
        Y_comp = Y_main - self.main_stream(X_main)
        Y_comp[:, :-increase_size] = 0

        # Train the compensation stream
        X_comp = self.activation_comp(X)
        self.comp_stream.fit(X_comp, Y_comp)

    @torch.no_grad()
    def update(self) -> None:
        self.main_stream.update()
        self.comp_stream.update()


class DSALLearner(ACILLearner):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.gamma_comp = args["gamma_comp"]
        self.compensation_ratio = args["compensation_ratio"]
        super().__init__(args, backbone, backbone_output, device, all_devices)

    def make_model(self) -> None:
        self.model = DSAL(
            self.backbone_output,
            self.backbone,
            self.buffer_size,
            self.gamma,
            self.gamma_comp,
            self.compensation_ratio,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )
