import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.models.components.attention import SelfAttentionMLP


class TransformerContextCombiner(eqx.Module):
    agent_proj: eqx.nn.Linear
    map_proj: eqx.nn.Linear | None
    tl_proj: eqx.nn.Linear | None
    rel_proj: eqx.nn.Linear | None
    layers: list
    out_dim: int

    def __init__(
        self,
        agent_dim: int,
        out_dim: int,
        hidden_dim: int,
        key,
        map_dim: int = 0,
        tl_dim: int = 0,
        rel_dim: int = 0,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        ap, mp, tp, rp, lk = jr.split(key, 5)
        layer_keys = jr.split(lk, num_layers)

        self.out_dim = out_dim
        self.agent_proj = eqx.nn.Linear(agent_dim, out_dim, use_bias=False, key=ap)
        self.map_proj = eqx.nn.Linear(map_dim, out_dim, use_bias=False, key=mp) if map_dim > 0 else None
        self.tl_proj = eqx.nn.Linear(tl_dim, out_dim, use_bias=False, key=tp) if tl_dim > 0 else None
        self.rel_proj = eqx.nn.Linear(rel_dim, out_dim, use_bias=False, key=rp) if rel_dim > 0 else None

        def _make_zero_init_layer(lk):
            layer = SelfAttentionMLP(
                attn_dim=out_dim,
                attn_num_heads=num_heads,
                out_dim=out_dim,
                mlp_dim=hidden_dim,
                num_mlp_layers=2,
                drop_attn=0.0,
                key=lk,
            )
            # zero-init attn out_proj and MLP last layer → starts as identity
            layer = eqx.tree_at(
                lambda l: l.attn.output_proj.weight, layer, jnp.zeros_like(layer.attn.output_proj.weight)
            )
            layer = eqx.tree_at(
                lambda l: l.mlp.layers[-1].weight, layer, jnp.zeros_like(layer.mlp.layers[-1].weight)
            )
            return layer

        self.layers = [_make_zero_init_layer(lk) for lk in layer_keys]

    def __call__(
        self,
        agent_encodings,   # [A, agent_dim]
        agents_mask,       # [A] True=invalid
        scene_map=None,    # [map_dim] global pooled
        scene_tl=None,     # [tl_dim] global pooled
        scene_rel=None,    # [A, rel_dim] per-agent
    ):
        a = agent_encodings.shape[0]

        tokens = jax.vmap(self.agent_proj)(agent_encodings)  # [A, out_dim]
        if scene_rel is not None and self.rel_proj is not None:
            tokens = tokens + jax.vmap(self.rel_proj)(scene_rel)
        elif scene_rel is not None:
            tokens = tokens + scene_rel

        extra_tokens = []
        extra_mask = []
        if scene_map is not None and self.map_proj is not None:
            extra_tokens.append(self.map_proj(scene_map)[None])
            extra_mask.append(jnp.array([False]))
        if scene_tl is not None and self.tl_proj is not None:
            extra_tokens.append(self.tl_proj(scene_tl)[None])
            extra_mask.append(jnp.array([False]))

        if extra_tokens:
            all_tokens = jnp.concatenate([tokens] + extra_tokens, axis=0)
            all_mask = jnp.concatenate([agents_mask] + extra_mask, axis=0)
        else:
            all_tokens = tokens
            all_mask = agents_mask

        valid = ~all_mask
        attn_mask = valid[:, None] & valid[None, :]

        for layer in self.layers:
            all_tokens = layer(all_tokens, attn_mask=attn_mask)
            all_tokens = jnp.where(all_mask[:, None], 0.0, all_tokens)

        return all_tokens, all_mask  # [A+M+TL, out_dim], [A+M+TL]
