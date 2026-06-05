import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class RelationEncoder(eqx.Module):
    proj: eqx.nn.MLP

    def __init__(self, hidden_dim: int = 256, key=None):
        self.proj = eqx.nn.MLP(
            in_size=4,
            width_size=hidden_dim,
            depth=1,
            out_size=hidden_dim,
            key=key,
        )

    def __call__(self, relations, pair_mask):
        dx = relations[..., 0]
        dy = relations[..., 1]
        dtheta = relations[..., 2]
        rel_features = jnp.stack(
            [dx, dy, jnp.sin(dtheta), jnp.cos(dtheta)],
            axis=-1,
        )
        edge_emb = jax.vmap(jax.vmap(self.proj))(rel_features)
        edge_emb = jnp.where(pair_mask[..., None], edge_emb, 0.0)
        denom = jnp.maximum(pair_mask.sum(axis=-1, keepdims=True), 1)
        return edge_emb.sum(axis=1) / denom
