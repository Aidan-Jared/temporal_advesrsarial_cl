import jax
import jax.numpy as jnp

from jaxtyping import Array, PRNGKeyArray

def reservoir_sampling(
    sample_idx: Array,
    labels: Array,
    logits: Array,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    seen_examples: int,
    *,
    device,
    key: PRNGKeyArray,
):
    batch_size = sample_idx.shape[0]
    buffer_size = buffer_idx.shape[0]
    replace = []
    choices = []
    for i in range(batch_size):
        if seen_examples < buffer_size:
            replace.append(seen_examples)
            choices.append(i)
        else:
            key, subkey = jax.random.split(key)
            rand_idx = jax.random.randint(subkey, (), 0, seen_examples+1).item()

            if rand_idx < buffer_size:
                replace.append(rand_idx)
                choices.append(i)
            
        seen_examples += 1

    choices = jnp.array(choices, device = device, dtype=jnp.uint32)
    replace = jnp.array(replace, device = device, dtype=jnp.uint32)
    buffer_idx = buffer_idx.at[replace].set(sample_idx[choices])
    buffer_targets = buffer_targets.at[replace].set(labels[choices].astype(jnp.uint32))
    buffer_logits = buffer_logits.at[replace].set(logits[choices])
    
    return buffer_idx, buffer_targets, buffer_logits, seen_examples
