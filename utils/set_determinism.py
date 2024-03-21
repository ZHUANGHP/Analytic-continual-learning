# -*- coding: utf-8 -*-

import torch
import numpy
from os import environ
import random


def set_determinism(seed: int) -> None:
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
