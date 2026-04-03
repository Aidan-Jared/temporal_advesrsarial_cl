from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute, FlatData
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.training.plugins import SupervisedPlugin
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.pacol import PACOL
from tqdm import tqdm


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
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,  # type: ignore
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

        flat = FlatData([TensorDataset(all_x, all_y)], indices=None)
        # tensor_dataset.targets = all_y.tolist()

        tl_da = DataAttribute(all_tid, "targets_task_labels", use_in_getitem=True)
        t_da = DataAttribute(all_y.tolist(), "targets")

        return ClassificationDataset(
            datasets=[flat],
            data_attributes=[t_da, tl_da],
            transform_groups=dataset._flat_data._transform_groups,
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
        key: PRNGKeyArray,
    ) -> None:
        self.splits = splits
        self.batch_size = batch_size
        self.seen_tasks = []
        self.iter_device = iter_device

        self.len = getattr(dataset, "__len__", batch_size)

        class_to_indices = {}
        all_data = []
        for idx, (data, label) in enumerate(dataset):  # type: ignore
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
            jnp.full((self.num_classes, max_samples_per_class), -1, dtype=jnp.int32),
            device,
        )

        self.class_lengths = jax.device_put(
            jnp.zeros(self.num_classes, dtype=jnp.int32), device
        )

        for class_idx, (label, idx) in enumerate(sorted(class_to_indices.items())):
            num_samples = len(idx)
            self.class_indicies = self.class_indicies.at[class_idx, :num_samples].set(
                jnp.array(idx, dtype=jnp.int32)
            )

            self.class_lengths = self.class_lengths.at[class_idx].set(num_samples)

        self.tasks = np.arange(self.num_classes).reshape((self.splits, -1))

    def __len__(self) -> int:
        return self.len
    
    def iters(self, task_n: int) -> int:
        task_idx = self.tasks[task_n]
        n = jnp.sum(self.class_lengths[task_idx]).item()
        return n // self.batch_size
    
    def sample(self, task_n: int, *, key: PRNGKeyArray | None = None):

        task_idx = self.tasks[task_n]
        n = jnp.sum(self.class_lengths[task_idx]).item()
        class_idx = self.class_indicies[task_idx].reshape(-1)
        labels = np.stack(
            [jnp.full(self.class_lengths[i], fill_value=i) for i in task_idx]
        ).reshape(-1)

        if key is not None:
            shuffle = jax.random.permutation(key=key, x=n)
            class_idx = class_idx[shuffle]
            labels = labels[shuffle]

        batches = n // self.batch_size

        start_idx = 0
        end_idx = self.batch_size
        for i in range(batches):
            X = self.all_data[class_idx[start_idx:end_idx]]
            y = labels[start_idx:end_idx]

            device = jax.devices(self.iter_device)[0]
            X = jax.device_put(X, device)
            y = jax.device_put(y, device)

            yield (X, y)
            start_idx += self.batch_size
            end_idx += self.batch_size


def _step(params, static, x, y, state, optim, opt_state, loss_fn, *, key):
    model = eqx.combine(params, static)
    (loss, (acc, state)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y, state, key
    )
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    params, _ = eqx.partition(model, eqx.is_array)
    return params, loss, acc, state, opt_state


def eval(model, state, tasks, testloader, loss_fn, *, key):
    model = eqx.nn.inference_mode(model, value=True)
    loss_fn = eqx.filter_jit(loss_fn)
    results = dict()
    for p_task in range(tasks):
        key, subkey = jax.random.split(key)
        task_loss = []
        task_acc = []
        pbar = tqdm(
            enumerate(testloader.sample(p_task, key=subkey)),
            total=testloader.iters(p_task),
            ncols=100
        )
        for step, (x, y) in pbar:
            loss, (acc, _) = loss_fn(model, x, y, state, key)
            task_loss.append(loss)
            task_acc.append(acc)
            if step % 10 == 0:
                pbar.set_postfix({"task_eval": p_task, "loss": np.mean(loss).item(), "acc": np.mean(acc).item()})
        results[p_task] = {"loss": np.mean(task_loss).item(), "acc": np.mean(task_acc).item()}

    return results
