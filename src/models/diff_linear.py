from math import prod

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from src.data_module.agent_path import AgentPath


class DiffLinear(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc_out: eqx.nn.Linear
    out_shape: tuple[int, ...]

    def __init__(
        self,
        hid_dim: int,
        input_shape: list[int],
        denoise_shape: list[int],
        key,
        **kwargs,
    ):
        k1, k2, k3 = jr.split(key, 3)
        traj_dim, cond_dim = prod(denoise_shape), prod(input_shape)
        self.out_shape = denoise_shape
        self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
        self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

    def __call__(self, t_noise, x_t, past_path: AgentPath, **batch_kwargs):
        past_traj = past_path.to_local()[..., :2]
        past_rel = (past_traj - past_traj[:, :1, :]).reshape(-1)
        x = jnp.concatenate(
            [x_t.reshape(-1), past_rel, jnp.atleast_1d(t_noise)], axis=0
        )
        x = jnn.relu(self.fc1(x))
        x = jnn.relu(self.fc2(x))
        return self.fc_out(x).reshape(self.out_shape)
