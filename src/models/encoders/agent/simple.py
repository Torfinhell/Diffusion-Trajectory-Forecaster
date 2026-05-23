import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr


class SimpleAgentEncoder(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, time_len: int, num_feat: int, out_dim: int, key):
        self.mlp = eqx.nn.MLP(
            in_size=time_len * num_feat,
            width_size=out_dim,
            depth=1,
            out_size=out_dim,
            activation=jnn.relu,
            key=key,
        )

    def __call__(self, x):
        if x.ndim == 4:
            x = x[0]
        a = x.shape[0]
        return jax.vmap(self.mlp)(x.reshape(a, -1))
