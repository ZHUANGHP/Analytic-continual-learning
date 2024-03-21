# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from typing import Tuple, Iterable, Optional, Callable
from .metrics import ClassificationMeter


@torch.no_grad()
def validate(
    model: Callable[[torch.Tensor], torch.Tensor],
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    desc: Optional[str] = None
) -> ClassificationMeter:
    if isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device
    else:
        device = model.device
    meter = ClassificationMeter(num_classes)

    for X, y in tqdm(data_loader, desc=desc):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Calculate the loss
        logits: torch.Tensor = model(X)
        meter.record(y, logits)
    return meter
