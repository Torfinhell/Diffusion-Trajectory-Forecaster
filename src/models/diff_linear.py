from math import prod

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


class DiffLinear(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc_out: eqx.nn.Linear
    out_shape: tuple[int, ...]

    def __init__(
        self,
        hid_dim: int,
        input_shape: list[int],
        output_shape: list[int],
        key,
    ):
        k1, k2, k3 = jr.split(key, 3)
        traj_dim, cond_dim = prod(output_shape), prod(input_shape)
        self.out_shape = output_shape
        self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
        self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

    def __call__(self, t_noise, x_t, **batch_kwargs):
        cond = batch_kwargs.get("agent_past", batch_kwargs.get("cond", None))
        if cond is None:
            raise ValueError(
                "DiffLinear expects `agent_past` (or `cond`) in batch kwargs."
            )
        x = jnp.concatenate(
            [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t_noise)], axis=0
        )
        x = jnn.relu(self.fc1(x))
        x = jnn.relu(self.fc2(x))
        return self.fc_out(x).reshape(self.out_shape)
