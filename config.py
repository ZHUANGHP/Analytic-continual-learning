import argparse

from models import models
from typing import Any, Dict
from os import path, makedirs
from datasets import dataset_list
from datetime import datetime
import yaml
from sys import argv

__all__ = ["load_args"]

_parser = argparse.ArgumentParser(description="Analytic Continual Learning")

# Method Options
_parser.add_argument(
    "method",
    choices=["ACIL", "DS-AL", "GKEAL", "G-ACIL"],
    help="The method to use for continual learning.",
)

_parser.add_argument(
    "--exp-name",
    type=str,
    default="",
    help="Name of the experiment",
)

# Dataset settings
_data_group = _parser.add_argument_group("Dataset arguments")
_data_group.add_argument(
    "-d",
    "--dataset",
    default="CIFAR-100",
    choices=dataset_list.keys(),
)

_data_group.add_argument(
    "--data-root",
    metavar="DIR",
    type=str,
    help="Root path to the dataset",
    default="~/dataset",
)

_data_group.add_argument(
    "-j",
    "--num-workers",
    default=8,
    type=int,
    metavar="N",
    help="Number of data loading workers (default: 8)",
)

_data_group.add_argument(
    "--base-ratio",
    default=0.5,
    type=float,
    help="The ratio of base classes in the training set.",
)

_data_group.add_argument(
    "--phases",
    "--tasks",
    default=10,
    type=int,
    help="Number of incremental phases (tasks).",
)

_data_group.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="The size of one mini-batch per GPU.",
)

_data_group.add_argument(
    "--cache-features",
    action="store_true",
    help="Load the features extracted by the frozen backbone to speed up inference.",
)

# Model settings
_model_group = _parser.add_argument_group("Model arguments")
_model_group.add_argument(
    "-a",
    "-arch",
    "--backbone",
    type=str,
    default="resnet32",
    help="model to use for training",
    choices=models.keys(),
    metavar="ARCH",
)

_model_group.add_argument(
    "--cache-path",
    "--backbone-path",
    metavar="DIR",
    type=str,
    help=(
        "Path to the base pretrain backbone."
        "If file exists, the base training will be skipped."
    ),
)

# Training Settings
_model_group.add_argument("--seed", default=None, type=int, help="Seed for models.")

_model_group.add_argument(
    "--dataset-seed", default=None, type=int, help="Seed for shuffling the dataset."
)

# Base training arguments
_base_group = _parser.add_argument_group("Base training arguments")

_base_group.add_argument(
    "--base-epochs",
    default=300,
    type=int,
    metavar="N",
    help="Number of total epochs to run for base training.",
)

_base_group.add_argument(
    "--warmup-epochs",
    default=10,
    type=int,
    metavar="N",
    help="Number of warmup epochs.",
)

_base_group.add_argument(
    "-lr",
    "--learning-rate",
    default=0.5,
    type=float,
    metavar="LR",
    help="Initial learning rate",
)

_base_group.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="Momentum for SGD"
)

_base_group.add_argument(
    "--wd",
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    dest="weight_decay",
)

_base_group.add_argument(
    "--separate-decay",
    action="store_true",
    help="Separating the normalization parameters from the rest of the model parameters",
)

_base_group.add_argument("--label-smoothing", default=0.05, type=float)

# IL hyper-parameters
_il_group = _parser.add_argument_group("IL Hyper-parameters")

_il_group.add_argument(
    "--IL-batch-size",
    default=None,
    type=int,
    help="The size of mini-batch during the incremental learning process.",
)

_il_group.add_argument(
    "--gamma",
    "--gamma-main",
    default=0.1,
    type=float,
    help="The regularization of the (main stream) linear classifier.",
)

_il_group.add_argument(
    "--buffer-size",
    "--expansion-size",
    default=8192,
    type=int,
    help="The buffer size of the classifier.",
)

_il_group.add_argument(
    "--gamma-comp",
    default=0.1,
    type=float,
    help="The regularization of the linear classifier in compensation stream (DS-AL only)",
)

_il_group.add_argument(
    "--sigma",
    default=10,
    type=float,
    help="The width-adjusting of the Gaussian kernel (GKEAL only)",
)

_il_group.add_argument(
    "-C",
    "--compensation-ratio",
    default=1,
    type=float,
    help="The regularization of the linear classifier in compensation stream (DS-AL only)",
)


def load_args() -> Dict[str, Any]:
    global _parser
    args = vars(_parser.parse_args())
    args["data_root"] = path.expanduser(args["data_root"])
    if args["cache_path"] is not None:
        assert path.isdir(args["cache_path"]), "The cache path is not a directory."
        backbone_path = path.join(args["cache_path"], "backbone.pth")
        args["backbone_path"] = backbone_path if path.isfile(backbone_path) else None
    saving_root = path.join(
        "saved_models",
        f"{args['backbone']}_{args['dataset']}_{args['base_ratio']}_{args['dataset_seed']}",
    )
    args["exp_name"] = args["exp_name"].strip()
    if args["exp_name"] == "":
        args["exp_name"] = args["method"]
    saving_root = path.join(saving_root, args["exp_name"])

    if args["IL_batch_size"] is None:
        args["IL_batch_size"] = args["batch_size"]

    # Windows does not support ":" in the path
    current_time = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    saving_root = path.join(saving_root, current_time)
    args["saving_root"] = saving_root
    args["argv"] = str(argv)
    makedirs(saving_root, exist_ok=True)
    with open(path.join(saving_root, "args.yaml"), "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(args, yaml_file)
    args["data_root"] = path.join(args["data_root"], args["dataset"])
    return args


if __name__ == "__main__":
    print(load_args())
