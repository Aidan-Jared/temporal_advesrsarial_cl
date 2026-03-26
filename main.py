import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from src.utils import CL_DataLoader
from src.resnet import ResNet18
from src.cl_methods.ewc import (
    EWC_loss, 
    compute_importance, 
    update_importances
)

from typing import Callable
from equinox.nn._stateful import State
from jaxtyping import PRNGKeyArray, PyTree, Array

SEED = 42
KEY = jax.random.PRNGKey(SEED)

def init(
    model: eqx.Module,
    optim: optax.GradientTransformationExtraArgs,
    init_fn: Callable | None = None,
    *,
    key: PRNGKeyArray
    
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    state = eqx.nn.make_with_state(model)(key) #type: ignore
    
    if init_fn is None:
        return opt_state, state
    
    return opt_state, state, init_fn(model)
    

def train(
    model: eqx.Module,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    optim: optax.GradientTransformationExtraArgs,
    criterion: Callable,
    steps_p_task: int,
    tasks: int,
    print_every: int,
)->eqx.Module:
    
    # @eqx.filter_jit    
    def make_step(
        model: eqx.Module,
        x: Array,
        y: Array,
        state: State,
        opt_state: PyTree,
        *,
        key: PRNGKeyArray | None = None
    ):
        loss = EWC_loss(
            model,
            x,
            y,
            state,
            criterion,
            importances,
            saved_params,
            task,
            lambda_,
            key=key,
        )
    return model

def main():
    normalize_data = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train = CIFAR10(
        root = "/Data/",
        train = True,
        transform = normalize_data,
        download=True,
    )

    test = CIFAR10(
        root = "/Data/",
        train = False,
        transform = normalize_data,
        download=True,
    )
    
    subkey1, subkey2, subkey3 = jax.random.split(KEY, 3)
    
    trainloader = CL_DataLoader(train, batch_size=32, splits=10, key=subkey1)
    testloader = CL_DataLoader(test, batch_size=32, splits=10, key=subkey2)
    
    p_model = ResNet18(3, key=subkey3)
    # c_model = ResNet18(3, key=subkey3)
    optim = optax.adam(learning_rate=1e-3)
    criterion = optax.softmax_cross_entropy

if __name__ == "__main__":
    main()
