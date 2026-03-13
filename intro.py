from torch.optim import SGD
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive#, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC
from avalanche.training.plugins import GEMPlugin, ReplayPlugin
from avalanche.training import ReservoirSamplingBuffer
from avalanche.benchmarks.classic import SplitMNIST
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

from utils import PoisoningPlugin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleMLP(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=.001, momentum=.9)
criterion = CrossEntropyLoss()

def perturbation(x):
    return add_pattern_bd(x, pixel_value=1., channels_first=True)

test = PoisoningAttackBackdoor(perturbation=perturbation)

test_p = PoisoningPlugin(attack=test, poison_to=1)

gem = GEMPlugin(
    patterns_per_experience=10,
    memory_strength=.5
    )

buffer = ReservoirSamplingBuffer(max_size=10)

replay = ReplayPlugin(
    mem_size=10,
    storage_policy= buffer
)

cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=100, train_epochs=4, eval_mb_size=100,
    device=device,
    plugins=[test_p, replay],
)

benchmark = SplitMNIST(n_experiences=5, seed=1)



print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    # loader = DataLoader(experience.dataset, batch_size=32)
    # for x, y, tid in loader:
    #     x = x.detach().cpu().numpy()
    #     y = y.detach().cpu().numpy()
    #     x, y = test.poison(x=x, y=y)
    #     break
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(benchmark.test_stream))