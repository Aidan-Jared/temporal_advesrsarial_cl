import equinox as eqx
import jax
import optax
import polars as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from src.cl_methods.ewc import EWC_train
from src.resnet import ResNet18
from src.utils import CL_DataLoader

SEED = 42
KEY = jax.random.PRNGKey(SEED)


def main():
    normalize_data = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
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

    subkey1, subkey2, subkey3 = jax.random.split(KEY, 3)

    trainloader = CL_DataLoader(train, batch_size=32, splits=5, key=subkey1)
    testloader = CL_DataLoader(test, batch_size=32, splits=5, key=subkey2)

    # p_model = ResNet18(3, key=subkey3)
    p_model, state = eqx.nn.make_with_state(ResNet18)(3, key=subkey3)
    # c_model = ResNet18(3, key=subkey3)
    optim = optax.adam(learning_rate=1e-3)
    criterion = optax.softmax_cross_entropy_with_integer_labels

    p_model, results = EWC_train(
        p_model, state, trainloader, testloader, optim, criterion, 3, 5, 10, key=subkey1
    )

    df = pl.from_dicts(results)
    df.write_parquet("ewc_results.parquet")
    print(df)


if __name__ == "__main__":
    main()
