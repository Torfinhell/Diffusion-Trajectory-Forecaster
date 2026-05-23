import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class ContextCombiner(eqx.Module):
    """Zero-init additive fusion of per-agent encodings with scene tokens."""

    map_proj: eqx.nn.Linear
    tl_proj: eqx.nn.Linear
    rel_proj: eqx.nn.Linear

    def __init__(
        self,
        agent_dim: int,
        out_dim: int,
        hidden_dim: int,
        key,
        map_dim: int = 0,
        tl_dim: int = 0,
        rel_dim: int = 0,
    ):
        del hidden_dim

        def _zero_linear(in_dim, fold_id):
            lin = eqx.nn.Linear(
                max(int(in_dim), 1),
                out_dim,
                use_bias=False,
                key=jr.fold_in(key, fold_id),
            )
            return eqx.tree_at(
                lambda layer: layer.weight, lin, jnp.zeros_like(lin.weight)
            )

        self.map_proj = _zero_linear(agent_dim + map_dim, 0)
        self.tl_proj = _zero_linear(agent_dim + tl_dim, 1)
        self.rel_proj = _zero_linear(agent_dim + rel_dim, 2)

    def __call__(
        self,
        agent_encodings,
        agents_mask,
        scene_map=None,
        scene_tl=None,
        scene_rel=None,
    ):
        out = agent_encodings
        a = agent_encodings.shape[0]
        if scene_map is not None:
            inp = jnp.concatenate(
                [
                    agent_encodings,
                    jnp.broadcast_to(scene_map[None], (a, scene_map.shape[0])),
                ],
                axis=-1,
            )
            out = out + jax.vmap(self.map_proj)(inp)
        if scene_tl is not None:
            inp = jnp.concatenate(
                [
                    agent_encodings,
                    jnp.broadcast_to(scene_tl[None], (a, scene_tl.shape[0])),
                ],
                axis=-1,
            )
            out = out + jax.vmap(self.tl_proj)(inp)
        if scene_rel is not None:
            inp = jnp.concatenate([agent_encodings, scene_rel], axis=-1)
            out = out + jax.vmap(self.rel_proj)(inp)
        return jnp.where(agents_mask[:, None], 0.0, out)
