from functools import partial
import equinox as eqx
import jax
import optax
import pandas as pd

from src.cl_methods.ewc import EWC_train
from src.cl_methods.gem import GEM_train
from src.models.resnet18 import singleHeadResNet18, multiHeadResNet18, kaiming_init_model
from src.models.resnet32 import singleHeadResNet32, multiHeadResNet32
from src.utils import CL_DataLoader, poinson_images, load_data

import argparse
import os
import ast
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def parse_list(arg):
    return ast.literal_eval(arg)

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=parse_list, default=[42])
#ewc: .05, gem: 1e-4, agem: 1e-4
parser.add_argument("--lr", type=float, default=.05)
parser.add_argument("--batch_size", type=int, default=128)
#ewc: 5, gem: 1, agem: 1
parser.add_argument("--task_epochs", type=int, default=1)
parser.add_argument("--data_set", type=str, default="CIFAR10")
parser.add_argument("--task_splits", type=int, default=2)
parser.add_argument("--model", type=str, default="multiHeadResNet32")
parser.add_argument("--norm", type=parse_list, default=[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)])

# Dataset Mean Std
# MNIST (0.1307,) (0.3081,)
# FashionMNIST (0.2860,) (0.3530,)
# CIFAR-10 (0.4914, 0.4822, 0.4465) (0.2470, 0.2435, 0.2616)
# CIFAR-100 (0.5071, 0.4867, 0.4408)(0.2675, 0.2565, 0.2761)

parser.add_argument("--method", type=str, default="EWC", choices=["EWC", "GEM", "AGEM"])
parser.add_argument("--lambda_", type=float, default=5e3)
parser.add_argument("--mem_strength", type=float, default=.5)
parser.add_argument("--mem_size", type=int, default=256)

parser.add_argument("--poison_attacks", type=parse_list, default=["gaussian_noise","shot_noise"])
parser.add_argument("--poison_tasks", type=parse_list, default=[0])
parser.add_argument("--pcp", type=float, default=.5)
parser.add_argument("--pp", type=float, default=.5)

args = parser.parse_args()

methods = {
    "EWC": EWC_train,
    "GEM": partial(GEM_train, method_name="GEM"),
    "AGEM": partial(GEM_train, method_name="AGEM"),
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
    df = pd.DataFrame()
    for seed in args.seed:
        print(type(seed))
        KEY = jax.random.PRNGKey(seed)
        train, test = load_data(args.data_set)

        dtype = jax.numpy.float32
        subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(KEY, 5)
    
        if "multiHeadResNet" in args.model:
            multi_head = True
        else:
            multi_head = False
        
        trainloader = CL_DataLoader(
            train, batch_size=args.batch_size, splits=args.task_splits, dtype=dtype, key=subkey1, mutl_head=multi_head
        )
        
        
        # poision_loader = CL_DataLoader(
        #     train, batch_size=args.batch_size, splits=args.task_splits, dtype=dtype, key=subkey1, mutl_head=multi_head
        # )
        
        testloader = CL_DataLoader(
            test, batch_size=args.batch_size, splits=args.task_splits, dtype=dtype, key=subkey2
        )
        
        
        trainloader.normalize(args.norm[0], args.norm[1])
        # poision_loader.normalize(args.norm[0], args.norm[1])
        testloader.normalize(args.norm[0], args.norm[1])
        
        gem_keys = {'lambda_', 'mem_strength', 'mem_size', 'poison_attacks', 'poison_tasks', 'pcp', 'pp'}
        kwargs = {k: v for k, v in vars(args).items() if k in gem_keys}    
        
        # poision_loader = poinson_images(poision_loader, tasks=args.poison_tasks, pcp=args.pcp, pp=args.pp, corruption=args.poison_attacks, key=subkey3)
            
        p_model, state = eqx.nn.make_with_state(models[args.model])(
            trainloader.all_data[0].shape[0], num_classes = trainloader.num_classes, num_splits = args.task_splits, dropout=0.0, dtype=dtype, key=subkey4
        )
        p_model = kaiming_init_model(p_model, subkey4)
        
        # c_model = eqx.nn.make_with_state(ResNet18)(3, dtype=dtype, key=subkey4)
        optim = optax.sgd(learning_rate=args.lr, momentum=.9)
        criterion = optax.softmax_cross_entropy_with_integer_labels
    
        method = methods[args.method]
        
        p_model, results = method(
            p_model, state, trainloader, testloader, optim, criterion, args.task_epochs, args.task_splits, 1, key=subkey5, **kwargs
        )
        
        # p_model, results = EWC_train(
        #     p_model, state, trainloader, testloader, optim, criterion, args.task_epochs, args.task_splits, 1, key=subkey5
        # )
        # p_model, results = GEM_train(
        #     p_model, state, trainloader, testloader, optim, criterion, task_epochs=1, tasks=5, print_every=10, method_name = "AGEM", memory_strength=10, task_samples = 10, key=subkey1
        # )
        meta = {"seed": seed, "model": args.model, "method": args.method, "data_set": args.data_set, **kwargs}
        results = [{**r, **meta} for r in results]
        
        df = pd.concat([df, pd.DataFrame(results)])

    path = f"experiment_res/{args.method}_{args.model}_{args.data_set}.parquet"
    df.to_parquet(path)
    print(df)


if __name__ == "__main__":
    main()
