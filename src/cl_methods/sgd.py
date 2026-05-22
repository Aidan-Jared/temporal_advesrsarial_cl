import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx
from optax import softmax_cross_entropy_with_integer_labels

from src.dataloader import CL_DataLoader
from src.utils import _step, eval, model_forward
from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32
from tqdm import tqdm

from typing import Callable
from jaxtyping import PRNGKeyArray, Array


def loss_fn(model, x, y, state, key, task):
    keys = jax.random.split(key, x.shape[0])
    logits, state = jax.vmap(
        model_forward,
        in_axes=(None, 0, None, None, 0),
        out_axes=(0, None),
        axis_name="batch",
    )(model, x, state, task, keys)
    loss = softmax_cross_entropy_with_integer_labels(logits, y)
    pred_y = jnp.argmax(logits, axis=1)
    acc = jnp.mean(y == pred_y)
    return jnp.mean(loss), (acc, state)


def sgd_train(
    model: ResNet18 | ResNet32,
    state: eqx.nn.State,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    optim,
    criterion: Callable,
    task_epochs: int,
    tasks: int,
    print_every: int,
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    results = []
    for task in range(tasks):
        task_loss = []
        task_acc = []
        model = eqx.nn.inference_mode(model, value=False)

        if task > 0 and hasattr(model, "add_head"):
            model = model.add_head(
                len(trainloader.tasks[task]), key=key
            )  # typing: ignore

        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        # params, static = eqx.partition(model, eqx.is_inexact_array)
        print("training task: ", task, "-" * 10)

        for epoch in range(task_epochs):
            key, subkey = jax.random.split(key)
            pbar = tqdm(
                enumerate(trainloader.sample(task, key=subkey)),
                total=trainloader.iters(task),
                # ncols=75,
            )
            for step, (x, y, _, _, _) in pbar:
                loss_fnt = jax.tree_util.Partial(loss_fn, task=task)
                model, loss, acc, state, opt_state = eqx.filter_jit(_step)(
                    model,
                    x,
                    y,
                    state,
                    optim,
                    opt_state,
                    loss_fnt,
                    key=subkey,
                )
                task_loss.append(loss)
                task_acc.append(acc)

                if (step + 1) % (
                    trainloader.class_lengths[task]
                    // (trainloader.batch_size * print_every)
                ).item() == 0:
                    pbar.set_postfix(
                        {
                            "epoch": epoch + 1,
                            "task": task,
                            "batch": step + 1,
                            "loss": np.mean(task_loss),
                            "acc": np.mean(task_acc),
                        }
                    )

        print("eval", "-" * 10)
        res = eval(
            model,
            state,
            tasks,
            testloader,
            loss_fnt,
            key=key,
        )
        res["task_trained"] = task
        print(res)
        results.append(res)
    return model, results
