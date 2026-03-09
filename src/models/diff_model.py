import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from hydra.utils import instantiate

from .base_model import BaseDiffusionModel


class SceneEncoder(eqx.Module):
    pass


class DiffDenoiser(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc_out: eqx.nn.Linear

    def __init__(self, hid_dim: int, out_dim: int, history: int, key):
        k1, k2, k3 = jr.split(key, 3)
        # x_t: [H*2], cond: [history*2], t: scalar.
        in_dim = out_dim + 2 * history + 1
        self.fc1 = eqx.nn.Linear(in_dim, hid_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
        self.fc_out = eqx.nn.Linear(hid_dim, out_dim, key=k3)

    def __call__(self, t, x_t, cond):
        x = jnp.concatenate(
            [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
        )
        x = jnn.relu(self.fc1(x))
        x = jnn.relu(self.fc2(x))
        return self.fc_out(x)


class DiffusionModel(BaseDiffusionModel):
    def __init__(self, hid_dim, traj_len, num_blocks, history, **kwargs):
        self.hid_dim = hid_dim
        self.traj_len = traj_len
        self.num_blocks = num_blocks
        self.history = history
        super().__init__(**kwargs)

    def get_model(self, key_model):
        future_steps = self.traj_len - self.history
        return DiffDenoiser(
            out_dim=2 * future_steps,
            hid_dim=self.hid_dim,
            history=self.history,
            # num_blocks=self.num_blocks,
            key=key_model,
        )

    def configure_optimizers(self):
        self.optim = optax.adam(3e-4)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

    def configure_ddpm_scheduler(self):
        self.int_beta = lambda t: t
        self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
        self.t1 = 10.0
        self.dt0 = 0.1
