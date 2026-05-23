import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from src.models.components.batch import squeeze_batch


class MapEncoder(eqx.Module):
    point_in: eqx.nn.Linear
    point_out: eqx.nn.Linear
    traffic_light_embed: eqx.nn.Embedding
    type_embed: eqx.nn.Embedding

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128, key=None):
        point_in_key, point_out_key, tl_key, type_key = jr.split(key, 4)
        self.point_in = eqx.nn.Linear(3, hidden_dim, key=point_in_key)
        self.point_out = eqx.nn.Linear(hidden_dim, embed_dim, key=point_out_key)
        self.traffic_light_embed = eqx.nn.Embedding(8, embed_dim, key=tl_key)
        self.type_embed = eqx.nn.Embedding(21, embed_dim, key=type_key)

    def __call__(self, inputs):
        inputs = squeeze_batch(inputs, name="MapEncoder")
        if inputs.ndim != 3:
            raise ValueError(f"MapEncoder expected 3D input, got {inputs.shape}")

        point_features = jax.vmap(
            jax.vmap(lambda point: self.point_out(jnn.relu(self.point_in(point))))
        )(inputs[..., :3])
        output = jnp.max(point_features, axis=-2)
        traffic_light_type = jnp.clip(inputs[:, 0, 3].astype(jnp.int32), 0, 7)
        traffic_light_embed = jax.vmap(self.traffic_light_embed)(traffic_light_type)
        polyline_type = jnp.clip(inputs[:, 0, 4].astype(jnp.int32), 0, 20)
        type_embed = jax.vmap(self.type_embed)(polyline_type)
        return output + traffic_light_embed + type_embed
