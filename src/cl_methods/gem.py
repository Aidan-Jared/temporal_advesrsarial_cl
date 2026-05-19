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
from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32


def loss_fn(model, x, y, state, task, criterion, key):
    key, *keys = jax.random.split(key, x.shape[0]+1)
    keys = jnp.stack(keys)
    logits, state = jax.vmap(model_forward, in_axes=(None, 0, None, None, 0), out_axes = (0, None), axis_name = "batch")(
        model, x, state, task, keys
    )
    # if task is not None and task > 0:
    #     jax.debug.print("{}", jnp.argmax(logits, axis=1))
    #     jax.debug.breakpoint()
    loss = criterion(logits, y)
    loss = jnp.mean(loss)
    pred_y = jnp.argmax(logits, axis=1)
    acc = jnp.mean(y == pred_y)
    return loss, (acc, state)

@eqx.filter_jit
def get_gradients(
    model: eqx.Module,
    state: State,
    criterion: Callable,
    memory: dict[int, tuple[Array, Array]],
    *,
    key: PRNGKeyArray,
) -> PyTree:
    grad_list = [] 
    def step(model, x, y, state, t, key):
        loss, _ = loss_fn(model, x, y, state, t, criterion, key)
        return loss

    step = eqx.filter_grad(step)
    key, subkey = jax.random.split(key)
    for t in memory.keys():
        x, y = memory[t]
        # x = jnp.split(x, 8)
        # y = jnp.split(y, 8)
        # for x_batch, y_batch in zip(x,y):
        # device = jax.local_devices()[0]
        # x = jax.device_put(x, device=device)
        # y = jax.device_put(y, device=device)
        key, subkey = jax.random.split(key)
        grads = step(model, x, y, state, t, subkey)
        if hasattr(grads, "heads"):            
            grads = eqx.tree_at(lambda m: m.heads, grads, replace_fn=lambda x: None)
        grad_list.append(grads)
    return jax.tree.map(lambda *g: jnp.stack(g, axis=0), *grad_list)


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
    for (x, y) in data.sample(task, key=subkey):
        # device = jax.devices("cpu")[0]
        # x = jax.device_put(x, device=device)
        # y = jax.device_put(y, device=device)
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


def solve_qp(M: PyTree, grads: PyTree, memory_strength: float) -> PyTree:
    if hasattr(grads, "heads"):            
        grads = eqx.tree_at(lambda m: m.heads, grads, replace_fn=lambda x: None)
    g_leaves = jax.tree.leaves(grads)
    m_leaves = jax.tree.leaves(M)
    g_flat = jnp.concatenate([g.reshape(-1) for g in g_leaves])
    m_flat = jnp.concatenate([m.reshape(m.shape[0], -1) for m in m_leaves], axis=1)
    
    t = m_flat.shape[0]
    eye = jnp.eye(t)
    h = jnp.full(t, memory_strength)
    
    P = jnp.dot(m_flat, m_flat.T)
    P = 0.5 * (P + P.T) + 1e-3 * eye
    q = jnp.dot(m_flat, g_flat)
    v, *_ = qpax.solve_qp(
        Q=P, q=-q, h=-h, G=-eye.T, A=jnp.zeros((0, t)), b=jnp.zeros(0)
    )
    v_star = jnp.dot(v, m_flat) + g_flat
    sizes = [g.size for g in g_leaves]
    splits = np.cumsum(sizes[:-1]).tolist()
    v_star = jnp.split(v_star, splits)
    return jax.tree_util.tree_unflatten(jax.tree.structure(grads), [v_star[i].reshape(g_leaves[i].shape) for i in range(len(g_leaves))])
    
    # def solve_qp_single(m, g):
    #     shape = g.shape
    #     n = g.size
    #     t = m.shape[0]

    #     m = m.reshape(t, n)
    #     g = g.reshape(n)

    #     G = jnp.eye(t)
    #     h = jnp.full(t, memory_strength)
    #     P = jnp.dot(m, m.T)
    #     P = 0.5 * (P + P.T) + 1e-3 * G
    #     q = jnp.dot(m, g) * -1
    #     v, *_ = qpax.solve_qp(
    #         Q=P, q=-q, h=-h, G=-G.T, A=jnp.zeros((0, t)), b=jnp.zeros(0)
    #     )
    #     v_star = jnp.dot(v, m) + g
    #     return v_star.reshape(shape).astype(jnp.float32)

    # if hasattr(grads, "heads"):            
    #     grads = eqx.tree_at(lambda m: m.heads, grads, replace_fn=lambda x: None)
        
    # return jax.tree.map(solve_qp_single, M, grads)


def GEM(
    M: PyTree,
    grads: PyTree,
    memory_strength: float,
    task: int,
) -> PyTree:
    if task > 0:
        star = solve_qp(M, grads, memory_strength)
        # jax.debug.print("{}", jax.tree_util.tree_leaves(star))
        # jax.debug.breakpoint()
        return star
    else:
        return grads


def AGEM(
    M: PyTree,
    grads: PyTree,
    memory_strength: float,
    task: int,
) -> PyTree:
   if task == 0:
        return grads
   else:
        if hasattr(grads, "heads"):            
            grads = eqx.tree_at(lambda m: m.heads, grads, replace_fn=lambda x: None)
        g_leaves = jax.tree.leaves(grads)
        m_leaves = jax.tree.leaves(M)
        g_flat = jnp.concatenate([g.reshape(-1) for g in g_leaves])
        m_flat = jnp.concatenate([m.reshape(m.shape[0], -1) for m in m_leaves], axis=1)
        
        m_flat = jnp.mean(m_flat, axis=0)
        dotg = jnp.dot(g_flat, m_flat)
        alpha = dotg / jnp.dot(m_flat, m_flat)
        
        g_prog = jnp.where(dotg < 0, g_flat - alpha * m_flat, g_flat)
        
        sizes = [g.size for g in g_leaves]
        splits = np.cumsum(sizes[:-1]).tolist()
        g_prog = jnp.split(g_prog, splits)
        return jax.tree_util.tree_unflatten(jax.tree.structure(grads), [g_prog[i].reshape(g_leaves[i].shape) for i in range(len(g_leaves))])

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
    memory_strength,
    method_name,
    *,
    key,
): 

    (loss, (acc, state)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y, state, task, criterion, key
    )
    grads = method[method_name](G, grads, memory_strength, task)
    
    # jax.debug.print("{}", jax.tree_util.tree_leaves(grads))
    # jax.debug.breakpoint()
    
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    return model, state, opt_state, loss, acc

method = {
    "AGEM": AGEM,
    "GEM": GEM,
}

def GEM_train(
    model: ResNet18 | ResNet32,
    state: eqx.nn.State,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    optim,
    criterion: Callable,
    task_epochs: int,
    tasks: int,
    print_every: int,
    method_name: str = "GEM",
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    memory_strength = kwargs["mem_strength"]
    task_samples = kwargs["mem_size"]
    results = []
    memory: dict[int, tuple[jnp.ndarray, jnp.ndarray]] = dict()
    G = jax.tree.map(lambda p: jnp.zeros_like(p), eqx.filter(model, eqx.is_array))
    for task in range(tasks):
        task_loss = []
        task_acc = []
        
        if task > 0 and hasattr(model, "add_head"):
            model = model.add_head(len(trainloader.tasks[task]), key=key) #typing: ignore
            
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        model = eqx.nn.inference_mode(model, value=False)
        print("training task: ", task, "-" * 10)
        for epoch in range(task_epochs):
            key, subkey = jax.random.split(key)
            pbar = tqdm(
                enumerate(trainloader.sample(task, key=subkey)),
                total=trainloader.iters(task),
                # ncols=75,
            )

            for step, (X, y) in pbar:
                key, subkey = jax.random.split(key)
                if task > 0:
                    G = get_gradients(model, state, criterion, memory, key=subkey)
                
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
                    memory_strength,
                    method_name,
                    key = subkey,
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
                            "epoch": epoch + 1,
                            "batch": step + 1,
                            "loss": np.mean(task_loss),
                            "acc": np.mean(task_acc),
                        }
                    )
                    task_loss = []
                    task_acc = []

        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        
        print("eval", "-" * 10)
        
        loss_func = jax.tree_util.Partial(loss_fn, criterion = criterion)
        key, subkey = jax.random.split(key)
        
        res = eval(
            model, state, tasks, testloader, loss_func, key=subkey
        )
        results.append(res)
        
        memory = update_memory(trainloader, task, memory, task_samples, key=subkey1)
                
        jax.clear_caches()

    return model, results
