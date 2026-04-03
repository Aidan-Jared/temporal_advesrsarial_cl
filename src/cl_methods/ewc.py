
from tqdm import tqdm
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn._stateful import State
from jaxtyping import Array, PRNGKeyArray, PyTree

# from intro import criterion
from src.utils import CL_DataLoader, _step, eval


def compute_importance(
    model: eqx.Module,
    state: State,
    task_n: int,
    criteron: Callable,
    data: CL_DataLoader,
    batches: int,
    *,
    key: PRNGKeyArray,
) -> PyTree:
    model = eqx.nn.inference_mode(model, value=True)
    params, _ = eqx.partition(model, eqx.is_array)
    importance = jax.tree.map(lambda x: jnp.zeros(x.shape), params)
    key, subkey = jax.random.split(key)

    def step_fn(model, x, y, state, key, importance):
        def forward(model, x, state, key):
            logits, _ = model(x, state=state, key=key)
            return logits

        def loss_fn(model, x, y, state, key):
            logits = forward(model, x, state, key)
            loss = jax.nn.log_softmax(logits)

            return -loss[y]

        grads = jax.vmap(
            eqx.filter_grad(loss_fn),
            axis_name="batch",
            in_axes=(None, 0, 0, None, None),
        )(model, x, y, state, key)

        grads, _ = eqx.partition(grads, eqx.is_array)
        importance = jax.tree.map(lambda i, g: i + jnp.sum(g**2, axis=0), importance, grads)

        return importance

    step_fn = eqx.filter_jit(step_fn)
    for step, (x, y) in enumerate(data.sample(task_n, key=subkey)):
        importance = step_fn(model, x, y, state, key, importance)
        if step == batches - 1:
            break

    # imp_magnitude = jax.tree_util.tree_leaves(importance)
    # jax.debug.print("mean importance: {x}", x = imp_magnitude)
    # jax.debug.breakpoint()
    importance = jax.tree.map(
        lambda i: i / float(batches * data.batch_size), importance
    )
    return importance


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

    penalty = jnp.zeros(())

    def standard(penalty):
        for exp in range(task):
            saved_param = saved_params[exp]
            imp = importances[exp]

            
            penalty += jnp.sum(
                jnp.array(
                    jax.tree_util.tree_leaves(
                        jax.tree.map(
                            lambda i, c, s: jnp.sum(i * (c - s) ** 2),
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
    key: PRNGKeyArray,
    criteron: Callable,
    importances: dict[int, PyTree],
    saved_params: dict[int, PyTree],
    task: int,
    lambda_: float,
) -> tuple[Array, tuple[Array, State]]:

    key, *keys = jax.random.split(key, x.shape[0] + 1)
    keys = jnp.vstack(keys)

    def forward(x, state, key):
        logits, state = model(x, state=state, key=key)  # type: ignore
        return logits, state

    logits, state = jax.vmap(
        forward, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
    )(x, state, keys)

    loss = criteron(logits, y)
    params, _ = eqx.partition(model, eqx.is_array)
    loss = jnp.mean(loss) + lambda_ * ECW_penalty(
        importances, saved_params, params, task
    )

    pred_y = jnp.argmax(logits, axis=1)
    acc = jnp.mean(y == pred_y)
    return jnp.squeeze(loss), (acc, state)

def EWC_train(
    model: eqx.Module,
    state: eqx.nn.State,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    optim,
    lambda_: float,
    criterion: Callable,
    task_epochs: int,
    tasks: int,
    print_every: int,
    *,
    key: PRNGKeyArray,
):
    importances: dict[int, PyTree] = dict()
    saved_params: dict[int, PyTree] = dict()
    results = []

    for task in range(tasks):
        task_loss = []
        task_acc = []
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        model = eqx.nn.inference_mode(model, value=False)

        params, static = eqx.partition(model, eqx.is_array)
        print("training task: ", task,"-"*100)
        for _ in range(task_epochs):
            key, subkey = jax.random.split(key)
            pbar = tqdm(
                enumerate(trainloader.sample(task, key=subkey)),
                total=trainloader.iters(task),
                ncols=100
            )
            for (step, (x, y)) in pbar:
                loss_fn = jax.tree_util.Partial(
                    EWC_loss,
                    criteron=criterion,
                    importances=importances,
                    saved_params=saved_params,
                    task=task,
                    lambda_=lambda_,
                )

                key, subkey = jax.random.split(key)
                params, loss, acc, state, opt_state = eqx.filter_jit(
                    _step,  # static_argnames=["params", "static", "loss_fn"]
                )(params, static, x, y, state, optim, opt_state, loss_fn, key=subkey)

                task_loss.append(loss)
                task_acc.append(acc)
                
                if (step + 1) % (trainloader.class_lengths[task] // (trainloader.batch_size * print_every)).item() == 0:
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

        loss_fn = jax.tree_util.Partial(
            EWC_loss,
            criteron=criterion,
            importances=importances,
            saved_params=saved_params,
            task=task,
            lambda_=lambda_,
        )
        model = eqx.combine(params, static)
        print("eval", "-" * 100)
        res = eval(model, state, tasks, testloader, loss_fn, key=key)
        results.append(res)
        key, subkey = jax.random.split(key)
        importance = compute_importance(
            model, state, task, criterion, trainloader, 50, key=subkey
        )
        importances = update_importances(importance, importances, task)
        saved_params[task] = jax.tree.map(lambda x: x, params)

    return model, results
