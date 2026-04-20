from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn._stateful import State

from optax import softmax_cross_entropy_with_integer_labels
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import tqdm

from src.utils import CL_DataLoader, _step, eval, model_forward
from src.resnet import ResNet18


def compute_importance(
    model: eqx.Module,
    state: State,
    task_n: int,
    data: CL_DataLoader,
    batches: int,
    *,
    key: PRNGKeyArray,
) -> PyTree:
    model = eqx.nn.inference_mode(model, value=True)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    
    if hasattr(params, "heads"):
        params = eqx.tree_at(lambda m: m.heads, params, replace_fn=lambda x: None)        

    importance = jax.tree.map(lambda x: jnp.zeros(x.shape), params)
    key, subkey = jax.random.split(key)
    del params

    def step_fn(model, x, y, state, key, importance) -> PyTree:
        
        def loss_fn(model, x, y, state, key) -> Array:
            logits, _ = jax.vmap(model_forward, axis_name = "batch", in_axes=(None, 0, None, None, None), out_axes = (0, None))(model, x, state, task_n, key)
            # loss = jax.nn.log_softmax(logits)
            loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y))
            return loss

        grads = eqx.filter_grad(loss_fn)(model, x, y, state, key)

        if hasattr(grads, "heads"):
            grads = eqx.tree_at(lambda m: m.heads, grads, replace_fn=lambda x: None)        
            
        importance = jax.tree.map(
            lambda i, g: i + batches * g**2, importance, grads
        )
        

        return importance

    step_fn = eqx.filter_jit(step_fn)
    
    old_batch_size = data.batch_size
    data.update_batch_size(128)
    
    pbar = tqdm(
        enumerate(data.sample(task_n, key=subkey)),
        desc="fisher importance"
        # ncols=75,
    )
    for step, (x, y) in pbar:
        importance = step_fn(model, x, y, state, key, importance)
        # if step == batches - 1:
        #     break

    
    # imp_magnitude = jax.tree_util.tree_leaves(importance)
    # jax.debug.print("mean importance: {x}", x = imp_magnitude)
    # jax.debug.breakpoint()
    importance = jax.tree.map(
        lambda i: i / float(batches * data.batch_size), importance
    )
    
    data.update_batch_size(old_batch_size)
    
    return importance


def update_importances(
    new_importance: PyTree, importances: PyTree, task: int, alpha: float = .3
) -> PyTree:

    if task == 0:
        return new_importance
        
    return jax.tree.map(
        lambda n, o: (alpha * o + (1-alpha)*n),
        new_importance, 
        importances, 
    )


def ECW_penalty(
    importances: PyTree,
    saved_params: PyTree,
    current_param: PyTree,
    task: int,
) -> Array:

    penalty = jnp.zeros(())
    
    def standard(penalty):
        # for exp in range(task
        if hasattr(current_param, "heads"):
            c = eqx.tree_at(lambda m: m.heads, current_param, replace_fn=lambda x: None)
        else:
            c = current_param
        penalty = jnp.sum(
        jnp.array(
            jax.tree_util.tree_leaves(
                jax.tree.map(
                    lambda i, c, s: jnp.sum(i * (s - c) ** 2) / 2,
                    importances,
                    c,
                    saved_params,
                )
            )
        )
    )
        
        # imp_magnitude = jax.tree_util.tree_leaves(saved_params)
        # jax.debug.print("mean importance: {x}", x = imp_magnitude)
        # jax.debug.breakpoint()
        return penalty    
    
    if importances is None:
        return penalty
    else:
        return standard(penalty) 

    # return jax.lax.cond(
    #     task == 0, 
    #     lambda p: p, 
    #     standard, 
    #     penalty
    # )


def EWC_loss(
    model:ResNet18,
    x: Array,
    y: Array,
    state: State,
    key: PRNGKeyArray,
    criteron: Callable,
    importances: PyTree,
    saved_params: PyTree,
    task: int,
    lambda_: float,
) -> tuple[Array, tuple[Array, State]]:

    key, *keys = jax.random.split(key, x.shape[0] + 1)
    keys = jnp.vstack(keys)
    
    logits, state = jax.vmap(model_forward, in_axes=(None, 0, None, None, 0), out_axes=(0, None))(model, x, state, task, keys)

    loss = criteron(logits, y)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    loss = jnp.mean(loss) + lambda_ * ECW_penalty(
        importances, saved_params, params, task
    )

    pred_y = jnp.argmax(logits, axis=1)
    acc = jnp.mean(y == pred_y)
    return jnp.squeeze(loss), (acc, state)


def EWC_train(
    model: ResNet18,
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
    importances = None
    saved_params = None
    results = []
    

    for task in range(tasks):
        task_loss = []
        task_acc = []
        model = eqx.nn.inference_mode(model, value=False)
        
        if task > 0 and hasattr(model, "add_head"):
            model = model.add_head(len(trainloader.tasks[task]), key=key) #typing: ignore

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
            for step, (x, y) in pbar:
                loss_fn = jax.tree_util.Partial(
                    EWC_loss,
                    criteron=criterion,
                    importances=importances,
                    saved_params=saved_params,
                    task=task,
                    lambda_=lambda_,
                )

                key, subkey = jax.random.split(key)
                model, loss, acc, state, opt_state = eqx.filter_jit(
                    _step
                )(model, x, y, state, optim, opt_state, loss_fn, key=subkey)

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

        loss_fn = jax.tree_util.Partial(
            EWC_loss,
            criteron=criterion,
            importances=importances,
            saved_params=saved_params,
            task=None,
            lambda_=lambda_,
        )
        # model = eqx.combine(params, static)
        print("eval", "-" * 10)
        res = eval(model, state, tasks, testloader, loss_fn, key=key)
        results.append(res)
        key, subkey = jax.random.split(key)
        importance = compute_importance(
            model, state, task, trainloader, 20, key=subkey
        )
        importances = update_importances(importance, importances, task)
        
        params, static = eqx.partition(model, eqx.is_inexact_array)
        
        saved_params= jax.tree.map(jnp.array, params)
        if hasattr(saved_params, "heads"):            
            saved_params = eqx.tree_at(lambda m: m.heads, saved_params, replace_fn=lambda x: None)

        del params, static
        jax.clear_caches()

    return model, results
