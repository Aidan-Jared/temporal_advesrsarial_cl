from typing import Any

import torch
from torch.utils.data import Dataset

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
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute, FlatData
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.training.plugins import SupervisedPlugin
from torch.utils.data import DataLoader, TensorDataset

from src.pacol import PACOL

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array

import numpy as np


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
        dataset: AvalancheDataset,
        batch_size: None | int = None,
    ):
        collate_fn: Any | None = getattr(dataset, "collate_fn", None)

        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True #type: ignore
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

        flat = FlatData(
            [TensorDataset(all_x, all_y)],
            indices=None
        )
        # tensor_dataset.targets = all_y.tolist()

        tl_da = DataAttribute(all_tid, "targets_task_labels", use_in_getitem=True)
        t_da = DataAttribute(all_y.tolist(), "targets")

        return ClassificationDataset(
            datasets=[flat], data_attributes=[t_da, tl_da],
            transform_groups=dataset._flat_data._transform_groups
        )

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
                strategy.experience.dataset = self._poison_data(
                    strategy.experience.dataset
                )
        return strategy

class CL_DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        splits: int,
        device: str = "cpu",
        iter_device: str = "gpu",
        *,
        key: PRNGKeyArray
    ) -> None:
        self.key = key
        self.splits = splits
        self.batch_size = batch_size
        self.seen_tasks = []
        self.iter_device = iter_device
        
        self.len = getattr(dataset, "__len__", batch_size)
        
        class_to_indices = {}
        all_data = []
        for idx, (data, label) in enumerate(dataset): # type: ignore
            if isinstance(data, jnp.ndarray):
                all_data.append(np.array(data))
            else:
                all_data.append(data.numpy())
            label_int = int(label)
            if label_int not in class_to_indices:
                class_to_indices[label_int] = []
            
            class_to_indices[label_int].append(idx)
        
        device = jax.devices(device)[0]
        
        all_data_np = np.stack(all_data)
        
        self.all_data = jax.device_put(all_data_np, device)
        
        self.num_classes = len(class_to_indices)
        
        max_samples_per_class = max(len(v) for v in class_to_indices.values())
        
        self.class_indicies = jax.device_put(
            jnp.full((self.num_classes, max_samples_per_class), -1, dtype = jnp.int32),
            device
        )
        
        self.class_lenghts = jax.device_put(
            jnp.zeros(self.num_classes, dtype = jnp.int32), device
        )
        
        for class_idx, (label, idx) in enumerate(sorted(class_to_indices.items())):
            num_samples = len(idx)
            self.class_indicies = self.class_indicies.at[class_idx, :num_samples].set(
                jnp.array(idx, dtype=jnp.int32)
            )
            
            self.class_lengths = self.class_lenghts.at[class_idx].set(num_samples)
        
        self.tasks = np.arange(self.num_classes).reshape((-1, self.splits))
        
        self._sample_task_jit = jax.jit(
            self._sample_task_fn,
            static_argnames=["task_n", "batch_size", "class_p_task"]
        )
        
    
    def __len__(self) -> int:
        return self.len
    
    @staticmethod
    def _sample_task_fn(
        class_indicies: Array,
        all_data: Array,
        task_n: int,
        batch_size: int,
        tasks: Array,
        splits: int,
        *,
        key: PRNGKeyArray | None
    ) -> tuple[Array, Array]:
        
        def sample_class(
            carry: tuple[PRNGKeyArray, int],
            _
        ):
            key, idx = carry
            key, subkey = jax.random.split(key)
            class_row = class_indicies[idx]
            mask = class_row > 0
            valid_idx = class_row[jnp.where(mask, class_row, 0)]
            selected_idx = jax.random.choice(key = subkey, a=valid_idx, shape = (batch_size // splits,), replace=False)
            data = all_data[selected_idx]
            labels = jnp.full((batch_size // splits), tasks[task_n][idx], dtype=jnp.int32)
            
            return (key, idx +1), (data, labels)
        
        (key, _), (all_class_data, all_class_labels) = jax.lax.scan(
            sample_class,
            init=(key,0),
            length=len(tasks[task_n])
        )
        return all_class_data, all_class_labels    
    
    def sample(
        self,
        task_n: int
    ) -> tuple[Array, Array]:
        self.key, subkey1, subkey2 = jax.random.split(self.key,3)
        
        all_data, all_labels = self._sample_task_jit(
            self.class_indicies,
            self.all_data,
            task_n,
            self.batch_size,
            self.tasks,
            self.splits,
            subkey1
        )
        
        if self.iter_device == "gpu":
            device = jax.devices(self.iter_device)[0]
            all_data = jax.device_put(all_data, device)
            all_labels = jax.device_put(all_labels, device)
        shuffle_idx = jax.random.permutation(subkey2, all_data.shape[0])
        all_data = all_data[shuffle_idx]
        all_labels = all_labels[shuffle_idx]
        
        return all_data, all_labels