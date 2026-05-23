import equinox as eqx
import jax.numpy as jnp


class FourierEmbedding(eqx.Module):
    freqs: jnp.ndarray
    embed_dim: int

    def __init__(self, embed_dim, key):
        del key
        half = embed_dim // 2
        self.embed_dim = embed_dim
        self.freqs = jnp.exp(jnp.arange(half) * -(jnp.log(10000.0) / (half - 1)))

    def __call__(self, x):
        args = x * self.freqs
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)])
