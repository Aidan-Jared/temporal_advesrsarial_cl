from functools import partial
import equinox as eqx
import jax
import optax
import pandas as pd

from src.cl_methods.ewc import EWC_train
from src.cl_methods.gem import GEM_train
from src.cl_methods.der import train_der
from src.models.resnet18 import (
    singleHeadResNet18,
    multiHeadResNet18,
    kaiming_init_model,
)
from src.models.resnet32 import singleHeadResNet32, multiHeadResNet32
from src.utils import CL_DataLoader, poinson_images, load_data

import argparse
import os
import ast

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def parse_list(arg):
    return ast.literal_eval(arg)

def parse_bool(arg):
    return arg == "True"

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model-runs", type=int, default=5)
# ewc: .05, gem: 1e-4, agem: 1e-4
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=128)
# ewc: 5, gem: 1, agem: 1
parser.add_argument("--task_epochs", type=int, default=1)
parser.add_argument("--data_set", type=str, default="CIFAR10")
parser.add_argument("--task_splits", type=int, default=2)
parser.add_argument("--model", type=str, default="multiHeadResNet32")
parser.add_argument(
    "--norm",
    type=parse_list,
    default=[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
)

parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--transform", type=parse_bool, default="True")
parser.add_argument("--task-shuffle", type=parse_bool, default="False")

# Dataset Mean Std
# MNIST (0.1307,) (0.3081,)
# FashionMNIST (0.2860,) (0.3530,)
# CIFAR-10 (0.4914, 0.4822, 0.4465) (0.2470, 0.2435, 0.2616)
# CIFAR-100 (0.5071, 0.4867, 0.4408)(0.2675, 0.2565, 0.2761)

parser.add_argument(
    "--method", type=str, default="EWC", choices=["EWC", "GEM", "AGEM", "DER"]
)
parser.add_argument("--lambda_", type=float, default=5e3)
parser.add_argument("--alpha", type=float, default=0.5, help="for der and ewc")
parser.add_argument("--mem_strength", type=float, default=0.5)
parser.add_argument("--mem_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=600)
parser.add_argument("--replay_size", type=int, default=256)
parser.add_argument("--der-beta", type=float, default=0.5)
# parser.add_argument(
#     "--selection-method",
#     type=str,
#     default="reservoir_sampling",
#     choices=["reservoir_sampling", ""],
# )

parser.add_argument(
    "--poison_attacks", type=parse_list, default=["gaussian_noise", "shot_noise"]
)
parser.add_argument("--poison_tasks", type=parse_list, default=[0])
parser.add_argument("--pcp", type=float, default=0.5)
parser.add_argument("--pp", type=float, default=0.5)

args = vars(parser.parse_args())

methods = {
    "EWC": EWC_train,
    "GEM": partial(GEM_train, method_name="GEM"),
    "AGEM": partial(GEM_train, method_name="AGEM"),
    "DER": train_der,
}

models = {
    "singleHeadResNet18": singleHeadResNet18,
    "multiHeadResNet18": multiHeadResNet18,
    "singleHeadResNet32": singleHeadResNet32,
    "multiHeadResNet32": multiHeadResNet32,
}


# for hyperparameters
# https://github.com/ContinualAI/continual-learning-baselines/tree/main/experiments


def main():
    BATCH = args["batch_size"]
    SPLITS = args["task_splits"]
    LR = args["lr"]
    MOMENTUM = args["momentum"]
    EPOCHS = args["task_epochs"]

    train, test = load_data(args["data_set"])

    key = jax.random.PRNGKey(args["seed"])
    seeds = jax.random.randint(key, (args["model_runs"],), 1, 5000)
    df = pd.DataFrame()
    for seed in seeds:
        KEY = jax.random.PRNGKey(seed)

        dtype = jax.numpy.float32
        subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(KEY, 5)

        if not args["task_shuffle"]:
            subkey1 = None

        if args["method"] == "DER":
            buffer = True
            buffer_size = args["buffer_size"]
            replay_size = args["replay_size"]
        else:
            buffer = False
            buffer_size = 0
            replay_size = 0

        if "multiHeadResNet" in args["model"]:
            multi_head = True
        else:
            multi_head = False

        trainloader = CL_DataLoader(
            train,
            batch_size=BATCH,
            splits=SPLITS,
            dtype=dtype,
            key=subkey1,
            buffer=buffer,
            buffer_size=buffer_size,
            buff_size_mem=replay_size,
            multi_head=multi_head,
        )

        testloader = CL_DataLoader(
            test,
            batch_size=BATCH,
            splits=SPLITS,
            dtype=dtype,
            key=subkey1,
        )

        norm = args["norm"]
        trainloader.normilization_values(norm[0], norm[1])

        testloader.normilization_values(norm[0], norm[1])

        train_keys = {
            "lambda_",
            "mem_strength",
            "mem_size",
            "poison_attacks",
            "poison_tasks",
            "alpha",
            "der-beta",
            "pcp",
            "pp",
        }
        kwargs = {k: v for k, v in args.items() if k in train_keys}

        # poision_loader = poinson_images(poision_loader, tasks=args.poison_tasks, pcp=args.pcp, pp=args.pp, corruption=args.poison_attacks, key=subkey3)

        model, state = eqx.nn.make_with_state(models[args["model"]])(
            trainloader.all_data[0].shape[0],
            num_classes=trainloader.num_classes,
            num_splits=SPLITS,
            dropout=0.0,
            dtype=dtype,
            key=subkey4,
        )
        model = kaiming_init_model(model, subkey4)

        # c_model = eqx.nn.make_with_state(ResNet18)(3, dtype=dtype, key=subkey4)
        optim = optax.sgd(learning_rate=LR, momentum=MOMENTUM)
        criterion = optax.softmax_cross_entropy_with_integer_labels

        method = methods[args["method"]]

        model, results = method(
            model,
            state,
            trainloader,
            testloader,
            optim,
            criterion,
            EPOCHS,
            SPLITS,
            10,
            key=subkey5,
            **kwargs,
        )

        meta = {
            "seed": seed,
            "model": args["model"],
            "method": args["method"],
            "data_set": args["data_set"],
            **kwargs,
        }
        results = [{**r, **meta} for r in results]

        df = pd.concat([df, pd.DataFrame(results)])

        method_suffix = {
            "EWC": f"_lambda{args['lambda_']}",
            "GEM": f"_memstr{args['mem_strength']}",
            "AGEM": f"_memstr{args['mem_strength']}",
            "DER": f"_a{args['der_alpha']}_b{args['der_beta']}",
        }.get(args["method"], "")

        path = (
            f"experiment_res/"
            f"{args['method']}_"
            f"{args['model']}_"
            f"{args['data_set']}_"
            f"tasks{args['task_splits']}_"
            f"epochs{args['task_epochs']}_"
            f"lr{args['lr']}_"
            f"mem{args['mem_size']}_"
            f"seed{seed}_"
            f"{method_suffix}"
            f".parquet"
        )
        df.to_parquet(path)
    print(df)


if __name__ == "__main__":
    main()
