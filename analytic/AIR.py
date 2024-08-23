# -*- coding: utf-8 -*-
"""
Implementation of the AIR [1], an online exemplar-free generalized CIL approach on imbalanced datasets.

References:
[1] Fang, Di, et al.
    "AIR: Analytic Imbalance Rectifier for Continual Learning."
    arXiv preprint arXiv:2408.10349 (2024).
"""

import torch
from .ACIL import ACILLearner, ACIL
from .AnalyticLinear import GeneralizedARM

__all__ = ["AIR", "AIRLearner", "GeneralizedAIRLearner"]


class AIR(ACIL):
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        X = self.feature_expansion(X)
        self.analytic_linear.fit(X, y)


class AIRLearner(ACILLearner):
    def make_model(self) -> None:
        self.model = AIR(
            self.backbone_output,
            self.wrap_data_parallel(self.backbone),
            self.buffer_size,
            self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=GeneralizedARM,
        )


class GeneralizedAIRLearner(AIRLearner):
    pass
