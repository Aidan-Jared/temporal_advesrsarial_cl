from typing import Callable

import equinox as eqx
from equinox.nn._stateful import State
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from src.utils import CL_DataLoader


def compute_importance(
    model: eqx.Module,
    state: State,
    task_n: int,
    criteron: Callable,
    data: CL_DataLoader,
    batches: int,
    *,
    key: PRNGKeyArray | None = None,
) -> PyTree:
    params, _ = eqx.partition(model, eqx.is_array)
    importance = jax.tree.map(lambda x: jnp.zeros(x.shape), params)

    for _ in range(batches):
        x, y = data.sample(task_n)

        grads = eqx.filter_grad(criteron)(model, x, y, state, key)

        grads, _ = eqx.partition(grads, eqx.is_array)
        importances = jax.tree.map(lambda i, x: i + x**2, importance, grads)
        
    
    importances = jax.tree.map(lambda i: i / float(batches * data.batch_size), importance)

    return importances


def update_importances(
    importance: PyTree, importances: dict[int, PyTree], task: int
) -> PyTree:

    return {**importances, task: importance}


def ECW_penalty(
    importances: dict[int, PyTree],
    saved_params: dict[int, PyTree],
    current_param: PyTree,
    task: int,
) -> Array:

    penalty = jnp.zeros(1)

    def standard(penalty):
        for exp in range(task):
            saved_param = saved_params[exp]
            imp = importances[exp]

            penalty += jnp.sum(
                jnp.array(
                    jax.tree_util.tree_leaves(
                        jax.tree.map(
                            lambda i, c, s: i * (c - s) ** 2,
                            imp,
                            current_param,
                            saved_param,
                        )
                    )
                )
            )
        return penalty

    return jax.lax.cond(task == 0, lambda p: p, standard, penalty)


def EWC_loss(
    model: eqx.Module,
    x: Array,
    y: Array,
    state: State,
    criteron: Callable,
    importances: dict[int, PyTree],
    saved_params: dict[int, PyTree],
    task: int,
    lambda_: float,
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[PyTree, float]:

    def loss_fn(model):
        loss = criteron(model, x, y, state, key)
        params, _ = eqx.partition(model, eqx.is_array)
        loss = jnp.mean(loss) + lambda_ * ECW_penalty(
            importances, saved_params, params, task
        )
        return loss

    return eqx.filter_value_and_grad(loss_fn)(model)
