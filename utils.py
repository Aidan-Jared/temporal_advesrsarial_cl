import torch
from torch.utils.data import DataLoader, TensorDataset

from art.attacks.poisoning import perturbations
from art.attacks.poisoning import backdoor_attack, backdoor_attack_dgm, clean_label_backdoor_attack, adversarial_embedding_attack, gradient_matching_attack, bad_det

from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC
from avalanche.training.plugins import SupervisedPlugin
from avalanche.benchmarks.utils import as_taskaware_classification_dataset, AvalancheDataset

from avalanche.benchmarks.scenarios.deprecated.new_classes import NCExperience
from art.attacks.attack import PoisoningAttack

import numpy as np

# make plugins which are added to the strategies

class PoisoningPlugin(SupervisedPlugin):
    def __init__(
            self,
            attack: PoisoningAttack,
            poison_to: int = 1,
            percent_poison: float = .1,
            batch_size: int = 32
    ):
        '''
        attack
        '''
        super().__init__()
        self.attack = attack
        self.poison_to = poison_to
        self.percent_poison = percent_poison
        self.batch_size = batch_size

    def before_training_exp(self, strategy, **kwargs):

        experience = strategy.experience
        task_id = experience.task_label

        if self.poison_to is not None and task_id >= self.poison_to:
            return
        
        dataset = experience.dataset

        collate_fn = getattr(dataset, "collate_fn", None)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)

        all_x, all_y, all_tid = [], [], []

        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            poison_n = int(self.batch_size * self.percent_poison)

            if poison_n > 0:
                p_x = x[:poison_n].detach().cpu().numpy()
                p_y = y[:poison_n].detach().cpu().numpy()
                p_x, p_y = self.attack.poison(p_x, p_y)
                x[:poison_n] = torch.tensor(p_x, dtype=x.dtype)
                y[:poison_n] = torch.tensor(p_y, dtype=y.dtype)
            all_x.append(x)
            all_y.append(y)
            all_tid.append(tid)

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_tid = torch.cat(all_tid, dim=0)

        tensor_dataset = TensorDataset(all_x, all_y)
        tensor_dataset.targets = all_y.tolist()

        strategy.experience.dataset = as_taskaware_classification_dataset(tensor_dataset)

        return strategy