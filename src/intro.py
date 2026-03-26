import pandas as pd
import torch
import torch.nn as nn

# from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
# from art.attacks.poisoning.perturbations import add_pattern_bd
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import ReservoirSamplingBuffer
from avalanche.training.plugins import GEMPlugin, ReplayPlugin
from avalanche.training.supervised import (
    Naive,  # , CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC
)
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from src.pacol import PACOL
from src.utils import PoisoningPlugin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleMLP(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()


# def perturbation(x):
# return add_pattern_bd(x, pixel_value=1, channels_first=True)

p_model = SimpleMLP(num_classes=10).to(device)

test = PACOL(
    eps=1e-2,
    steps=3,
    step_size=1e-3,
    iterations=2,
    dist_metric="cosine",
    model=p_model,
    loss_fn=nn.CrossEntropyLoss(),
    opt=Adam(p_model.parameters(), lr=1e-3),
    batch_size=32,
)

test_p = PoisoningPlugin(attack=test, start_after=1, target_experiance=0)

gem = GEMPlugin(patterns_per_experience=10, memory_strength=0.5)

buffer = ReservoirSamplingBuffer(max_size=10)

replay = ReplayPlugin(mem_size=10, storage_policy=buffer)

poisoned_strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=100,
    train_epochs=4,
    eval_mb_size=100,
    device=device,
    plugins=[test_p, replay],
)

cl_strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=100,
    train_epochs=4,
    eval_mb_size=100,
    device=device,
    plugins=[replay],
)

benchmark = SplitMNIST(n_experiences=5, seed=1)


print("Starting experiment...")
results = {
    "poisoned": [],
    "clean": [],
}
for experience in benchmark.train_stream:
    # loader = DataLoader(experience.dataset, batch_size=32)
    # for x, y, tid in loader:
    #     x = x.detach().cpu().numpy()
    #     y = y.detach().cpu().numpy()
    #     x, y = test.poison(x=x, y=y)
    #     break
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    poisoned_strategy.train(experience)
    cl_strategy.train(experience)
    print("Training completed")

    print("Computing accuracy on the whole test set")
    results["clean"].append(cl_strategy.eval(benchmark.test_stream))

    results["poisoned"].append(poisoned_strategy.eval(benchmark.test_stream))


print("Results:")

print(pd.DataFrame.from_dict(results["poisoned"][-1], orient="index"))
print(pd.DataFrame.from_dict(results["clean"][-1], orient="index"))
