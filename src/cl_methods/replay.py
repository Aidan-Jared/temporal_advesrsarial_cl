import jax
import jax.numpy as jnp
# import numpy as np
from jaxtyping import Array, PRNGKeyArray
from src.utils import CL_DataLoader

def random_selection(
    data: CL_DataLoader,  
    task_n: int,
    n_samples: int,
    *,
    key: PRNGKeyArray,
):
    task_idx = data.tasks[task_n]
    class_idx = data.class_indicies[task_idx].reshape(-1)
    labels = jnp.repeat(task_idx, data.class_lengths[task_idx])
    idx = jax.random.permutation(key, jnp.sum(data.class_lengths[task_idx]))[:n_samples]
    sampled_data = data.all_data[class_idx[idx]]
    sampled_labels = labels[idx]
    return sampled_data, sampled_labels