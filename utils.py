from typing import Any

import torch

# from art.attacks.attack import PoisoningAttack
# from art.attacks.poisoning import (
#     adversarial_embedding_attack,
#     backdoor_attack,
#     backdoor_attack_dgm,
#     bad_det,
#     clean_label_backdoor_attack,
#     gradient_matching_attack,
#     perturbations,
# )
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    as_taskaware_classification_dataset,
)
from avalanche.benchmarks.utils.dataset_definitions import (
    ISupportedClassificationDataset,
)
from avalanche.training.plugins import SupervisedPlugin
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pacol import PACOL


class PoisoningPlugin(SupervisedPlugin):
    """
    applies a posioning attack to the provided strategy
    """

    def __init__(
        self,
        attack: PACOL,
        start_after: int = 1,
        pcp: float = 0.5,
        pn: int = 1,
        pp: float = 0.5,
        target_experiance: int = 1,
    ):
        """
        attack: type of attack to apply
        poison_to: task to stop poisioning at
        pcp: poisoned class percentage, how many classes to poison from the task
        pn: number of poisoning methods to use
        pp: percent of data to poision
        """
        super().__init__()
        self.attack: PACOL = attack
        self.start_after: int = start_after
        self.pcp: float = pcp
        self.pn: int = pn
        self.pp: float = pp
        self.restore: bool = True
        self.target_training: bool = False
        self.target_grad: bool = True
        self.target_experiance: int = target_experiance
        self.flip_data = None

    def _poison_data(
        self,
        dataset: Dataset,
        batch_size: None | int = None,
    ):
        collate_fn: Any | None = getattr(dataset, "collate_fn", None)

        if batch_size is None:
            batch_size: int = len(dataset)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )

        all_x, all_y, all_tid = [], [], []

        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            poison_n = int(batch_size * self.pp)

            if poison_n > 0:
                p_x = x[:poison_n].detach()
                p_y = y[:poison_n].detach()
                labels = torch.unique(p_y)
                for idx, c in enumerate(p_y):
                    if c == labels[0]:
                        p_y[idx] = labels[1]
                    else:
                        p_y[idx] = labels[0]

                p_x, p_y = self.attack(self.flip_data, [p_x, p_y])
                x[:poison_n] = p_x
                y[:poison_n] = p_y
            all_x.append(x)
            all_y.append(y)
            all_tid.append(tid)

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_tid = torch.cat(all_tid, dim=0)

        tensor_dataset = TensorDataset(all_x, all_y)
        tensor_dataset.targets = all_y.tolist()

        return as_taskaware_classification_dataset(tensor_dataset)

    def before_training_exp(self, strategy, *args, **kwargs) -> Any:
        if not self.target_training:
            return

        experience = strategy.experience

        if (
            self.start_after is not None
            and strategy.clock.train_exp_counter <= self.start_after
        ):
            return

        dataset = experience.dataset

        strategy.experience.dataset = self._poison_data(dataset)
        return strategy

    def after_training_exp(self, strategy, *args, **kwargs) -> Any:
        if not self.target_grad:
            return

        # dataset = strategy.experience.dataset
        t = strategy.clock.train_exp_counter
        # batch_size = strategy.train_mb_size
        if self.start_after is not None and t <= self.start_after:
            if self.flip_data is None and t == self.target_experiance:
                self.flip_data = strategy.experience.dataset
            return

        if t <= self.start_after:
            return

        for p in strategy.plugins:
            if hasattr(p, "memory_x"):
                device = p.memory_x[t].device
                p_x = p.memory_x[t].detach().cpu().numpy()
                p_y = p.memory_y[t].detach().cpu().numpy()
                poison_n = int(p_x.shape[0] * self.pp)

                p_x, p_y = self.attack(p_x[:poison_n], p_y[:poison_n])

                p.memory_x[t][:poison_n] = torch.tensor(
                    p_x, dtype=p.memory_x[t].dtype, device=device
                )
                p.memory_y[t][:poison_n] = torch.tensor(
                    p_y, dtype=p.memory_y[t].dtype, device=device
                )
            elif hasattr(p, "buffer"):
                strategy.experience.dataset = self._poison_data(
                    strategy.experience.dataset
                )
            elif hasattr(p, "storage_policy"):
                strategy.adapted_dataset = self._poison_data(
                    strategy.experience.dataset
                )
        return


# backdoor_attack_dgm()
