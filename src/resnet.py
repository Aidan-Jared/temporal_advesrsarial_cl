import os

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

SEED = 43
KEY = jax.random.PRNGKey(SEED)


class Drop_Path(eqx.Module):
    p: float
    inference: bool

    def __init__(self, p: float, inference: bool = False):
        self.p = p
        self.inference = inference

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
    ):

        def _drop(x, key):
            key_prob = 1 - self.p
            B = x.shape[0]
            shape = (B,) + (1,) * (x.ndim - 1)

            mask = jax.random.bernoulli(key, key_prob, shape)

            output = (x * mask) / key_prob

            return output

        return jax.lax.cond(
            self.p > 0.0 and not self.inference and key is None,
            _drop,
            lambda x, k: x,
            x,
            key,
        )


class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    shortcut: list | None

    bn1: eqx.nn.GroupNorm
    bn2: eqx.nn.GroupNorm
    dropout: Drop_Path

    def __init__(
        self,
        in_chanels: int,
        out_chanels: int,
        stride: int = 1,
        dropout: float = 0.1,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_chanels,
            out_chanels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dtype=dtype,
            key=subkey1,
        )
        self.conv2 = eqx.nn.Conv2d(
            out_chanels, out_chanels, kernel_size=3, stride=1, padding=1, dtype=dtype, key=subkey2
        )

        self.bn1 = eqx.nn.GroupNorm(8,out_chanels)
        self.bn2 = eqx.nn.GroupNorm(8,out_chanels)
        self.dropout = Drop_Path(dropout)

        if stride != 1 or in_chanels != out_chanels:
            self.shortcut = [
                eqx.nn.Conv2d(
                    in_chanels,
                    out_chanels,
                    kernel_size=1,
                    stride=stride,
                    use_bias=False,
                    dtype=dtype,
                    key=subkey3,
                ),
                eqx.nn.GroupNorm(8, out_chanels),
            ]
        else:
            self.shortcut = None

    def __call__(
        self,
        x: Float[Array, " batch c w h"],
        state: PyTree,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, PyTree]:

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.elu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = jax.nn.elu(out)

        out = self.dropout(out, key=key)

        if self.shortcut is not None:
            identity = self.shortcut[0](x)
            identity, state = self.shortcut[1](identity, state)
        else:
            identity = x

        # def _shortcut(x, state):
        #     i = self.shortcut[0](x)
        #     i, s = self.shortcut[1](i, state) 
        #     return i, s

        # identity, state = jax.lax.cond(
        #     self.shortcut is not None, _shortcut, lambda x, s: (x, s), x, state

        # )

        out = out + identity

        out = jax.nn.elu(out)
        return out, state


class ResNet18(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.GroupNorm
    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    layer4: eqx.nn.Sequential

    fc: eqx.nn.Linear

    hidden_channels: int

    avgpool: eqx.nn.AdaptiveAvgPool2d

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 10,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 6)
        self.hidden_channels = hidden_channels

        self.conv1 = eqx.nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            key=subkey1,
        )
        self.bn1 = eqx.nn.GroupNorm(8, hidden_channels)

        self.layer1 = self._make_layer(
            hidden_channels, num_blocks=2, stride=1, dtype=dtype, key=subkey2
        )
        self.layer2 = self._make_layer(
            hidden_channels * 2, num_blocks=2, stride=2, dtype=dtype, key=subkey3
        )
        self.layer3 = self._make_layer(
            hidden_channels * 4, num_blocks=2, stride=2, dtype=dtype, key=subkey4
        )
        self.layer4 = self._make_layer(
            hidden_channels * 8, num_blocks=2, stride=2, dtype=dtype, key=subkey5
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = eqx.nn.Linear(hidden_channels * 8, num_classes, key=subkey6)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int, dtype, key: PRNGKeyArray
    ):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides:
            key, subkey = jax.random.split(key)
            layers.append(
                BasicBlock(self.hidden_channels, out_channels, stride, dtype=dtype, key=subkey)
            )
            self.hidden_channels = out_channels
        return eqx.nn.Sequential(layers)

    def __call__(
        self,
        x: Float[Array, "batch c w h"],
        state: PyTree,
        *,
        key: PRNGKeyArray,
    ):
        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.elu(out)

        for block in self.layer1:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)
        for block in self.layer2:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)
        for block in self.layer3:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)
        for block in self.layer4:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)

        out = self.avgpool(out)
        out = out.reshape(
            out.shape[0]
        )

        out = self.fc(out)

        return out, state
