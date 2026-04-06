from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import qpax
from equinox.nn._stateful import State
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import tqdm

from src.utils import CL_DataLoader, eval, model_forward


def get_gradients(
    model: eqx.Module,
    state: State,
    criterion: Callable,
    task: int,
    memory: dict[int, tuple[Array, Array]],
    *,
    key: PRNGKeyArray,
) -> PyTree:
    if task == 0:
        return {}
    params, _ = eqx.partition(model, eqx.is_array)
    G = jax.tree.map(lambda p: [], params)
    del params

    def step(model, x, y, state, key):
        y_pred, _ = jax.vmap(
            model_forward, in_axes=(None, 0, None, 0), out_axes=(0, None)
        )(model, x, state, key)
        loss = criterion(y_pred, y)
        return loss

    step = eqx.filter_grad(eqx.filter_jit(step))
    key, subkey = jax.random.split(key)
    for t in range(task):
        x, y = memory[t]
        key, subkey = jax.random.split(key)
        grads = step(model, x, y, state, subkey)
        G = jax.tree.map(lambda g, p: g.append(p), G, grads)

    return jax.tree.map(lambda g: jnp.stack(g), G)


def update_memory(
    data: CL_DataLoader,
    task: int,
    memory: dict[int, tuple[Array, Array]],
    task_samples: int,
    *,
    key: PRNGKeyArray,
) -> dict[int, tuple[Array, Array]]:
    key, subkey = jax.random.split(key)
    size = 0
    for idx, (x, y) in enumerate(data.sample(task, key=subkey)):
        if x.shape[0] + size < task_samples:
            if task not in memory:
                memory[task] = (x, y)
            else:
                memory[task] = (
                    jnp.concatenate([memory[task][0], x]),
                    jnp.concatenate([memory[task][1], y]),
                )
        else:
            diff = task_samples - size
            x = x[:diff]
            y = y[:diff]
            if task not in memory:
                memory[task] = (x, y)
            else:
                memory[task] = (
                    jnp.concatenate([memory[task][0], x]),
                    jnp.concatenate([memory[task][1], y]),
                )
            break

        size += x.shape[0]
    return memory


def solve_qp(M: PyTree, grads: PyTree, memory_strenght: float) -> PyTree:
    def solve_qp_single(m, g):
        t = m.shape[0]
        G = jnp.eye(t)
        h = jnp.full(t, memory_strenght)
        P = jnp.dot(m, m.T)
        P = 0.5 * (P + P.T) + 1e-3 * G
        q = jnp.dot(m, g)
        v, *_ = qpax.solve_qp(
            Q=P, q=-q, h=h, G=-G.T, A=jnp.zeros((0, t)), b=jnp.zeros(0)
        )
        v_star = jnp.dot(v, m) + g
        return v_star.astype(jnp.float32)

    return jax.tree.map(solve_qp_single, M, grads)


def GEM(
    M: PyTree,
    grads: PyTree,
    memory_strenght: float,
    task: int,
) -> PyTree:
    return jax.lax.cond(
        task > 0,
        lambda m, g: solve_qp(m, g, memory_strenght),
        lambda m, g: g,
        M,
        grads,
    )


def AGEM(
    M: PyTree,
    grads: PyTree,
    memory_strenght: float,
    task: int,
) -> PyTree:
    def _agem(m, g):
        def _single(m, g):
            dotg = jnp.dot(g, m)
            alpha = dotg / jnp.dot(m, m)
            proj = g - m * alpha
            return proj

        return jax.tree.map(_single, m, g)

    return jax.lax.cond(
        task > 0,
        lambda m, g: _agem(m, g),
        lambda m, g: g,
        M,
        grads,
    )


method = {
    "AGEM": AGEM,
    "GEM": GEM,
}


@eqx.filter_jit
def train_step(
    model,
    x,
    y,
    state,
    criterion,
    G,
    optim,
    opt_state,
    task,
    memory_strenght,
    method_name,
    key,
):
    def loss_fn(model, state):
        logits, state = jax.vmap(model_forward, in_axes=(None, 0, 0, None, None))(
            model, x, state, key
        )
        loss = criterion(logits, y)
        loss = jnp.mean(loss)
        pred_y = jnp.argmax(logits, axis=1)
        acc = jnp.mean(y == pred_y)
        return loss, (acc, state)

    (loss, (acc, state)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, state
    )
    grads = method[method_name](G, grads, memory_strenght, task)
    model, opt_state = optim.update(grads, opt_state, model)

    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, loss, acc


def GEM_train(
    model: eqx.Module,
    state: eqx.nn.State,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    optim,
    criterion: Callable,
    task_epochs: int,
    tasks: int,
    print_every: int,
    method_name: str,
    memory_strenght: float,
    task_samples: int,
    *,
    key: PRNGKeyArray,
):
    results = []
    memory: dict[int, tuple[jnp.ndarray, jnp.ndarray]] = dict()
    G = jax.tree.map(lambda p: jnp.zeros_like(p), eqx.partition(model, eqx.is_array)[0])
    for task in range(tasks):
        task_loss = []
        task_acc = []
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        for epoch in range(task_epochs):
            key, subkey = jax.random.split(key)
            print("training task: ", task, "-" * 10)
            pbar = tqdm(
                enumerate(trainloader.sample(task, key=subkey)),
                total=trainloader.iters(task),
                ncols=75,
            )
            for step, (X, y) in pbar:
                key, *keys = jax.random.split(key, X.shape[0] + 1)
                keys = jnp.stack(keys)
                model, state, opt_state, loss, acc = train_step(
                    model,
                    X,
                    y,
                    state,
                    criterion,
                    G,
                    optim,
                    opt_state,
                    task,
                    memory_strenght,
                    method_name,
                    keys,
                )

                task_loss.append(loss)
                task_acc.append(acc)

                if (step + 1) % (
                    trainloader.class_lengths[task]
                    // (trainloader.batch_size * print_every)
                ).item() == 0:
                    pbar.set_postfix(
                        {
                            "task_train": task,
                            "batch": step + 1,
                            "loss": np.mean(task_loss),
                            "acc": np.mean(task_acc),
                        }
                    )
                    task_loss = []
                    task_acc = []

            key, subkey1, subkey2 = jax.random.split(key, 3)
            memory = update_memory(trainloader, task, memory, task_samples, key=subkey1)
            G = get_gradients(model, state, criterion, task, memory, key=subkey2)
            print("eval", "-" * 10)
            res = eqx.filter_jit(eval)(
                model, state, tasks, testloader, criterion, key=key
            )
            results.append(res)
            jax.clear_caches()

    return model, results
