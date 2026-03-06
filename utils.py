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


class PoisoningPlugin(SupervisedPlugin):
    '''
    applies a posioning attack to the provided strategy
    '''

    def __init__(
            self,
            attack: PoisoningAttack,
            poison_to: int = 1,
            pcp: float = .5,
            pn: int = 1,
            pp: float = .5,
    ):
        '''
        attack: type of attack to apply
        poison_to: task to stop poisioning at
        pcp: poisoned class percentage, how many classes to poison from the task
        pn: number of poisoning methods to use
        pp: percent of data to poision
        '''
        super().__init__()
        self.attack = attack
        self.poison_to = poison_to
        self.pcp = pcp
        self.pn = pn
        self.pp = pp
        self.restore = True
        self.target_training = False
        self.target_grad = True
        self.original_data = None

    def _poison_data(self, dataset, batch_size = None):
        if self.restore:
            self.original_data = dataset

        collate_fn = getattr(dataset, "collate_fn", None)

        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        all_x, all_y, all_tid = [], [], []

        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            poison_n = int(batch_size * self.pp)

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

        return as_taskaware_classification_dataset(tensor_dataset)

    
    def before_training_exp(self, strategy, **kwargs):
        if not self.target_training:
            return

        experience = strategy.experience

        if self.poison_to is not None and strategy.clock.train_exp_counter >= self.poison_to:
            return
        
        dataset = experience.dataset

        strategy.experience.dataset = self._poison_data(dataset)
        return strategy

    def after_training_exp(self, strategy, **kwargs):
        if not self.target_grad:
            return
        
        if self.original_data is not None:
            strategy.experience.dataset = self.original_data
            self.original_data = None

        # dataset = strategy.experience.dataset
        t = strategy.clock.train_exp_counter
        # batch_size = strategy.train_mb_size
        if self.poison_to is not None and t >= self.poison_to:
            return

        for p in strategy.plugins:
            if hasattr(p, "update_memory"):
                device = p.memory_x[t].device
                p_x = p.memory_x[t].detach().cpu().numpy()
                p_y = p.memory_y[t].detach().cpu().numpy()
                poison_n = int(p_x.shape[0] * self.pp)

                p_x, p_y = self.attack.poison(p_x[:poison_n], p_y[:poison_n])

                p.memory_x[t][:poison_n] = torch.tensor(p_x, dtype=p.memory_x[t].dtype, device=device)
                p.memory_y[t][:poison_n] = torch.tensor(p_y, dtype=p.memory_y[t].dtype, device=device)
        return 

    

# backdoor_attack_dgm()