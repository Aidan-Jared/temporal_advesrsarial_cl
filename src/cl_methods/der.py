import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, PRNGKeyArray
import equinox as eqx
import numpy as np

from tqdm import tqdm

from equinox.nn._stateful import State
from optax import (
    softmax_cross_entropy_with_integer_labels,
    GradientTransformationExtraArgs,
)

from src.dataloader import CL_DataLoader
from src.utils import model_forward, eval
from typing import Callable


def der_loss(
    Model,
    x: Array,
    y: Array,
    state: State,
    old_logits: Array,
    batch_size: int,
    der_alpha: float = 0.5,
    beta: float = 0.0,
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

    loss = softmax_cross_entropy_with_integer_labels(
        logits[:batch_size], y[:batch_size]
    )
    sloss = jax.lax.cond(
        jnp.all(old_logits) != 0.0,
        lambda: der_alpha * jnp.mean((logits[batch_size:] - old_logits) ** 2),
        lambda: 0.0,
    )
    loss += sloss
    if beta != 0.0:
        sloss = jax.lax.cond(
            jnp.all(old_logits) != 0.0,
            lambda: beta
            * jnp.mean(
                softmax_cross_entropy_with_integer_labels(
                    logits[batch_size:], y[batch_size:]
                )
            ),
            lambda: 0.0,
        )
        # jax.debug.print("{}", y[batch_size:])
        # jax.debug.breakpoint()
        loss += sloss
    return jnp.mean(loss), (acc, state) 

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
    *,
    key: PRNGKeyArray,
):
    (loss, (acc, state)), grads = eqx.filter_value_and_grad(
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
        key=key,
    )
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))

    model = eqx.apply_updates(model, updates)

    return model, loss, acc, state, opt_state


def train_der(
    model,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    tasks: int,
    epochs: int,
    state: State,
    optim: GradientTransformationExtraArgs,
    der_alpha: float = 0.5,
    beta: float = 0.0,
    selection_method: Callable | None = None,
    print_every: int = 10,
    *,
    key: PRNGKeyArray,
):
    batch_size = trainloader.batch_size
    results = []
    train_step_jit = eqx.filter_jit(train_step)
    for task in range(tasks):
        model = eqx.nn.inference_mode(model, False)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
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
                key, subkey = jax.random.split(key)
                model, loss, acc, state, opt_state = (
                    train_step_jit(
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
                        key=subkey,
                    )
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
                logits, _ = jax.vmap(
                    model_forward_jit,
                    in_axes=(None, 0, None, None),
                    out_axes=(0, None),
                    axis_name="batch",
                )(model, x, state, key)

                eval_loss.append(softmax_cross_entropy_with_integer_labels(logits, y))
                eval_acc.append(jnp.mean(jnp.argmax(logits, axis=1) == y))
                if (step + 1) == testloader.iters(task):
                    print("eval loss: ", np.mean(eval_loss))
                    print("eval acc: ", np.mean(eval_acc))
                    print()

        print("eval")
        print("-" * 50)
        res = eval(model, state, tasks, testloader, softmax_cross_entropy_with_integer_labels, key=subkey)

        results.append(res)
        if task < tasks - 1:
            trainloader.add_to_buffer(
                task,
                model,
                state,
                selection_method=selection_method,
                key=subkey,
            )
    return results
