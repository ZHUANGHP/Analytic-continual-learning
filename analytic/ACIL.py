# -*- coding: utf-8 -*-
"""
Implementation of the ACIL [1] and the G-ACIL [2].
The G-ACIL is a generalization of the ACIL in the generalized setting.
For the popular setting, the G-ACIL is equivalent to the ACIL.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from .Buffer import RandomBuffer
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from .AnalyticLinear import AnalyticLinear, RecursiveLinear

# 目前仅测试了Resnet32
# 采用相同的环境运行：
# python main.py ACIL --dataset CIFAR-100 --base-ratio 0.5 --phases 25 \
#     --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet32 \
#     --gamma 0.1 --buffer-size 8192 \
#     --backbone-path ./backbones/resnet32_CIFAR-100_0.5_None

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        out = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        out1 = self.backbone.layer1(out)
        out2 = self.backbone.layer2(out1)
        out3 = self.backbone.layer3(out2)
        pooled = F.avg_pool2d(out3, out3.size()[3])
        flat = pooled.view(pooled.size(0), -1)
        final_output = self.backbone.fc(flat)
        return out1, out2, out3, final_output


class ACIL(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 8192,
        gamma: float = 1e-3,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.feature_keys = ['layer1','layer2','layer3', 'final_output']
        # self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.buffer = {
            'layer1': RandomBuffer(16, buffer_size, **factory_kwargs),
            'layer2': RandomBuffer(32, buffer_size, **factory_kwargs),
            'layer3': RandomBuffer(64, buffer_size, **factory_kwargs),
            'final_output': RandomBuffer(64, buffer_size, **factory_kwargs),
        }
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.eval()

    @torch.no_grad()
    def feature_expansion(self, X: torch.Tensor) -> torch.Tensor:
        # print("Xshape------------------",X.shape)
        model1 = FeatureExtractor(self.backbone)
        out1, out2, out3, final_output = model1(X)
        # print("out1.shape:",out1.shape)
        # print("out2.shape:",out2.shape)
        # print("out3.shape:",out3.shape)
        # print("final.shape:",final_output.shape)
        expanded_features = []

        expanded_features.append(self.buffer['layer1'](torch.flatten(F.avg_pool2d(out1,out1.size()[3]),1)))
        # print("out1")
        # print(out1.view(out1.size(0),-1).shape)
        
        expanded_features.append(self.buffer['layer2'](torch.flatten(F.avg_pool2d(out2,out2.size()[3]),1)))
        # print("out2")
        # print(out2.view(out2.size(0),-1).shape)

        expanded_features.append(self.buffer['layer3'](torch.flatten(F.avg_pool2d(out3,out3.size()[3]),1)))
        # print("out3")
        # print(out3.view(out3.size(0),-1).shape)

        expanded_features.append(self.buffer['final_output'](final_output))
        # print("final_out")
        # return torch.cat(expanded_features,dim=1)
        
        # 用 torch.stack 堆叠特征，并计算它们的平均值
        stacked_features = torch.stack(expanded_features, dim=0)
        mean_features = torch.mean(stacked_features, dim=0)
        return mean_features

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.analytic_linear(self.feature_expansion(X))

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        Y = torch.nn.functional.one_hot(y)
        X = self.feature_expansion(X)
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()


class ACILLearner(Learner):
    """
    This implementation is for the G-ACIL [2], a general version of the ACIL [1] that
    supports mini-batch learning and the general CIL setting.
    In the traditional CIL settings, the G-ACIL is equivalent to the ACIL.
    """

    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.make_model()

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        model = torch.nn.Sequential(
            self.backbone,
            torch.nn.Linear(self.backbone_output, baseset_size),
        ).to(self.device, non_blocking=True)
        model = self.wrap_data_parallel(model)

        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()
        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6 # type: ignore
        )
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
            )

        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.args["label_smoothing"]
        ).to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
                model.train()
                for X, y in tqdm(train_loader, "Training"):
                    X: torch.Tensor = X.to(self.device, non_blocking=True)
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(X)
                    loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            train_meter = validate(
                model, train_loader, baseset_size, desc="Training (Validation)"
            )
            print(
                f"loss: {train_meter.loss:.4f}",
                f"acc@1: {train_meter.accuracy * 100:.3f}%",
                f"acc@5: {train_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
                sep="    ",
            )

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                if epoch != 0:
                    self.save_object(
                        (self.backbone, X.shape[1], self.backbone_output),
                        "backbone.pth",
                    )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.accuracy5,
                val_meter.f1_micro,
                train_meter.loss,
                train_meter.accuracy,
                train_meter.accuracy5,
                train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        self.backbone.eval()
        self.make_model()

    def make_model(self) -> None:
        self.model = ACIL(
            self.backbone_output,
            self.wrap_data_parallel(self.backbone),
            self.buffer_size,
            self.gamma,
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
        self.model.eval()
        for X, y in tqdm(data_loader, desc=desc):
            X: torch.Tensor = X.to(self.device, non_blocking=True)
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            self.model.fit(X, y, increase_size=incremental_size)

    def before_validation(self) -> None:
        self.model.update()

    def inference(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model
