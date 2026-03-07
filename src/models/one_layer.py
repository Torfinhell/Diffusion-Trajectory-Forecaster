import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

class TinyDenoiser(eqx.Module):
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
