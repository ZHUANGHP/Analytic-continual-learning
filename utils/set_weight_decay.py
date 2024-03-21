# -*- coding: utf-8 -*-
from gc import disable
import torch
from typing import Dict, List, Any


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    disable_norm_decay: bool = True,
    disable_bias_decay: bool = True,
    disable_embedding_decay: bool = True,
):
    # See: https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    params = {
        "other": [],
        "norm": [],
        "bias": [],
        "class_token": [],
        "position_embedding": [],
        "relative_position_bias_table": [],
    }

    params_weight_decay = {
        "bias": 0 if disable_bias_decay else weight_decay,
        "class_token": 0 if disable_embedding_decay else weight_decay,
        "position_embedding": 0 if disable_embedding_decay else weight_decay,
        "relative_position_bias_table": 0 if disable_embedding_decay else weight_decay,
    }

    def _add_params(module: torch.nn.Module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            for key in params_weight_decay.keys():
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    break
            else:
                if isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    params_weight_decay["other"] = weight_decay
    params_weight_decay["norm"] = 0.0 if disable_norm_decay else weight_decay

    param_groups: List = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups
