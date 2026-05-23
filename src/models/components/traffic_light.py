import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.models.components.batch import squeeze_batch


class TrafficLightEncoder(eqx.Module):
    type_embed: eqx.nn.Embedding

    def __init__(self, embed_dim: int = 256, key=None):
        self.type_embed = eqx.nn.Embedding(8, embed_dim, key=key)

    def __call__(self, inputs):
        inputs = squeeze_batch(inputs, name="TrafficLightEncoder")
        if inputs.ndim != 2:
            raise ValueError(
                f"TrafficLightEncoder expected 2D input, got {inputs.shape}"
            )
        traffic_light_type = jnp.clip(inputs[:, 2].astype(jnp.int32), 0, 7)
        return jax.vmap(self.type_embed)(traffic_light_type)
