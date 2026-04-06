import equinox as eqx
import jax
import optax
import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from src.cl_methods.ewc import EWC_train
from src.cl_methods.gem import GEM_train
from src.resnet import ResNet18
from src.utils import CL_DataLoader, poinson_images

SEED = 42
KEY = jax.random.PRNGKey(SEED)
# for hyperparameters
# https://github.com/ContinualAI/continual-learning-baselines/tree/main/experiments

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
    subkey1, subkey2, subkey3 = jax.random.split(KEY, 3)

    trainloader = CL_DataLoader(train, batch_size=32, splits=5, dtype=dtype, key=subkey1, replay=True)
    testloader = CL_DataLoader(test, batch_size=32, splits=5, dtype=dtype, key=subkey2)
    trainloader.normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    testloader.normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    # trainloader = poinson_images(
    #     trainloader, tasks=1, pcp=0.1, pp=0.1, severity=1, corruption=["fog", "shot_noise"], key=subkey3
    # )
    p_model, state = eqx.nn.make_with_state(ResNet18)(3, dtype=dtype, key=subkey3)
    # c_model = eqx.nn.make_with_state(ResNet18)(3, dtype=dtype, key=subkey3)
    optim = optax.adam(learning_rate=1e-4)
    criterion = optax.softmax_cross_entropy_with_integer_labels

    p_model, results = EWC_train(
        p_model, state, trainloader, testloader, optim, 3e9, criterion, 3, 5, 10, key=subkey1
    )
    # p_model, results = GEM_train(
    #     p_model, state, trainloader, testloader, optim, criterion, task_epochs=1, tasks=5, print_every=10, method_name = "AGEM", memory_strength=10, task_samples = 10, key=subkey1
    # )

    df = pd.DataFrame(results)
    df.to_parquet("ewc_results.parquet")
    print(df)


if __name__ == "__main__":
    main()
