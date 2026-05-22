from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn._stateful import State
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import (
    GradientTransformationExtraArgs,
    softmax_cross_entropy_with_integer_labels,
)
from tqdm import tqdm

from src.dataloader import CL_DataLoader
from src.utils import eval, model_forward
from src.buffer_selection import reservoir_sampling


def der_loss(
    Model,
    x: Array,
    y: Array,
    state: State,
    old_logits: Array,
    batch_size: int,
    der_alpha: float = 0.5,
    beta: float = 0.0,
    buffer_filled: bool = False,
    *,
    key: PRNGKeyArray,
):
    key, *keys = jax.random.split(key, x.shape[0] + 1)
    keys = jnp.array(keys)

    logits, state = jax.vmap(
        model_forward,
        in_axes=(None, 0, None, 0),
        out_axes=(0, None),
        axis_name="batch",
    )(Model, x, state, keys)

    acc = jnp.mean(jnp.argmax(logits, axis=1) == y)

    loss = jnp.mean(
        softmax_cross_entropy_with_integer_labels(logits[:batch_size], y[:batch_size])
    )
    if beta != 0.0:
        loss += jax.lax.cond(
            buffer_filled,
            lambda: der_alpha
            * jnp.mean(
                (
                    logits[batch_size : batch_size + old_logits.shape[0] // 2]
                    - old_logits[: old_logits.shape[0] // 2]
                )
                ** 2
            ),
            lambda: 0.0,
        )
        # jax.debug.print("{}", sloss)
        loss += jax.lax.cond(
            buffer_filled,
            lambda: beta
            * jnp.mean(
                softmax_cross_entropy_with_integer_labels(
                    logits[batch_size + old_logits.shape[0] // 2 :],
                    y[batch_size + old_logits.shape[0] // 2 :],
                )
            ),
            lambda: 0.0,
        )
        # jax.debug.print("{}", y[batch_size:])
        # jax.debug.breakpoint()
    else:
        loss += jax.lax.cond(
            buffer_filled,
            lambda: der_alpha * jnp.mean((logits[batch_size:] - old_logits) ** 2),
            lambda: 0.0,
        )
    # jax.debug.breakpoint()
    return loss, (logits, acc, state)  # typing: ignore


def train_step(
    model,
    x: Array,
    y: Array,
    state: State,
    old_logits: Array,
    batch_size: int,
    optim: GradientTransformationExtraArgs,
    opt_state: PyTree,
    der_alpha: float = 0.5,
    beta: float = 0.0,
    buffer_filled: bool = False,
    *,
    key: PRNGKeyArray,
):
    (loss, (logits, acc, state)), grads = eqx.filter_value_and_grad(
        der_loss, has_aux=True
    )(
        model,
        x,
        y,
        state,
        old_logits,
        batch_size,
        der_alpha,
        beta,
        buffer_filled,
        key=key,
    )
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))

    model = eqx.apply_updates(model, updates)

    return model, logits, loss, acc, state, opt_state


def train_der(
    model,
    state: State,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    optim: GradientTransformationExtraArgs,
    criterion: Callable,  # for compatiability
    epochs: int,
    tasks: int,
    print_every: int = 10,
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    der_alpha = kwargs.get("alpha", 0.5)
    beta = kwargs.get("der_beta", 0.5)
    batch_size = trainloader.batch_size
    results = []
    train_step_jit = eqx.filter_jit(train_step)
    for task in range(tasks):
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        model = eqx.nn.inference_mode(model, False)
        print(f"training task {task}")
        print("-" * 50)
        for epoch in range(epochs):
            key, subkey = jax.random.split(key)

            epoch_loss = []
            epoch_acc = []

            pbar = tqdm(
                enumerate(trainloader.sample(task, key=subkey)),
                total=trainloader.iters(task),
            )  # train_step_jit = train_step
            for step, (x, y, indexes, task_n, old_logits) in pbar:
                key, subkey1, subkey2 = jax.random.split(key, 3)
                buffer_filled = jnp.any(trainloader.buffer_idx >= 0).item() and task > 0
                model, logits, loss, acc, state, opt_state = train_step_jit(
                    model,
                    x,
                    y,
                    state,
                    old_logits,
                    batch_size,
                    optim,
                    opt_state,
                    der_alpha,
                    beta,
                    buffer_filled,
                    key=subkey1,
                )

                trainloader.add_to_buffer(
                    indexes[:batch_size],
                    y[:batch_size],
                    logits[:batch_size],
                    selection_method=reservoir_sampling,
                    key=subkey2,
                )

                epoch_loss.append(loss)
                epoch_acc.append(acc)
                if (step + 1) % print_every == 0:
                    pbar.set_postfix(
                        {
                            "task_train": task,
                            "epoch": epoch + 1,
                            "batch": step + 1,
                            "loss": np.mean(epoch_loss),
                            "acc": np.mean(epoch_acc),
                        }
                    )
                    epoch_loss = []
                    epoch_acc = []

            print("task eval")

            model_forward_jit = eqx.filter_jit(model_forward)
            eval_acc = []
            eval_loss = []
            model = eqx.nn.inference_mode(model, True)
            for step, (x, y, indexes, task_n, old_logits) in enumerate(
                testloader.sample(task, key=subkey)
            ):
                key, *keys = jax.random.split(key, x.shape[0] + 1)
                keys = jnp.stack(keys)
                logits, _ = jax.vmap(
                    model_forward_jit,
                    in_axes=(None, 0, None, None, 0),
                    out_axes=(0, None),
                    axis_name="batch",
                )(model, x, state, task, keys)

                eval_loss.append(softmax_cross_entropy_with_integer_labels(logits, y))
                eval_acc.append(jnp.mean(jnp.argmax(logits, axis=1) == y))
                if (step + 1) == testloader.iters(task):
                    print("eval loss: ", np.mean(eval_loss))
                    print("eval acc: ", np.mean(eval_acc))
                    print()

        print("eval")
        print("-" * 50)
        res = eval(
            model,
            state,
            tasks,
            testloader,
            softmax_cross_entropy_with_integer_labels,
            key=subkey,
        )

        res["task_trained"] = task

        results.append(res)
    return model, results
