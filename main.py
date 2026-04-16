from functools import partial
import equinox as eqx
import jax
import optax
import pandas as pd
from pandas.core.indexes.base import IgnoreRaise
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from src.cl_methods.ewc import EWC_train
from src.cl_methods.gem import GEM_train
from src.resnet import ResNet18
from src.utils import CL_DataLoader, poinson_images

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--task_epochs", type=int, default=3)
parser.add_argument("--data_set", type=str, default="CIFAR10")
parser.add_argument("--task_splits", type=int, default=5)
parser.add_argument("--model", type=str, default="ResNet18")
parser.add_argument("--norm", type=list, default=[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)])

parser.add_argument("--method", type=str, default="EWC", choices=["EWC", "GEM", "AGEM"])
parser.add_argument("--lambda", type=float, default=1e9)
parser.add_argument("--mem_strength", type=float, default=1.0)

parser.add_argument("--poison_attacks", type=list, default=["gaussian_noise","shot_noise"])
parser.add_argument("--poison_tasks", type=list, default=[0])
parser.add_argument("--pcp", type=float, default=.5)
parser.add_argument("--pp", type=float, default=.5)

args = parser.parse_args()

methods = {
    "EWC": EWC_train,
    "GEM": partial(GEM_train, method_name="GEM"),
    "AGEM": partial(GEM_train, method_name="AGEM"),
}

KEY = jax.random.PRNGKey(args.seed)
# for hyperparameters
# https://github.com/ContinualAI/continual-learning-baselines/tree/main/experiments
# LOOK AT HOW THE KEYS PROPOGATE

def main():
    normalize_data = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train = CIFAR10(
        root="Data/",
        train=True,
        transform=normalize_data,
        download=True,
    )

    test = CIFAR10(
        root="Data/",
        train=False,
        transform=normalize_data,
        download=True,
    )

    dtype = jax.numpy.float32
    subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(KEY, 5)

    trainloader = CL_DataLoader(
        train, batch_size=args.batch_size, splits=args.task_splits, dtype=dtype, key=subkey1 
    )
    
    
    poision_loader = CL_DataLoader(
        train, batch_size=args.batch_size, splits=args.task_splits, dtype=dtype, key=subkey1
    )
    
    testloader = CL_DataLoader(
        test, batch_size=args.batch_size, splits=args.task_splits, dtype=dtype, key=subkey2
    )
    
    
    trainloader.normalize(args.norm[0], args.norm[1])
    poision_loader.normalize(args.norm[0], args.norm[1])
    testloader.normalize(args.norm[0], args.norm[1])
    
    poision_loader = poinson_images(poision_loader, tasks=args.poision_tasks, pcp=args.pcp, pp=args.pp, key=subkey3)
        
    p_model, state = eqx.nn.make_with_state(ResNet18)(3, dtype=dtype, key=subkey4)
    # c_model = eqx.nn.make_with_state(ResNet18)(3, dtype=dtype, key=subkey4)
    optim = optax.adam(learning_rate=args.lr)
    criterion = optax.softmax_cross_entropy_with_integer_labels

    p_model, results = EWC_train(
        p_model, state, trainloader, testloader, optim, 3e9, criterion, 3, 5, 10, key=subkey5
    )
    # p_model, results = GEM_train(
    #     p_model, state, trainloader, testloader, optim, criterion, task_epochs=1, tasks=5, print_every=10, method_name = "AGEM", memory_strength=10, task_samples = 10, key=subkey1
    # )

    df = pd.DataFrame(results)
    df.to_parquet("experiment_res/ewc_results.parquet")
    print(df)


if __name__ == "__main__":
    main()
