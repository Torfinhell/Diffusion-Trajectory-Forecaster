import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

class _PreNormSelfAttentionMLP(eqx.Module):
    """Pre-norm residual block: x = x + sublayer(norm(x)).

    Unlike the shared post-norm `SelfAttentionMLP` (x = norm(x + sublayer(x))),
    pre-norm lets a zero-initialized sublayer output make the whole block an
    exact identity at init (x + 0 == x); LayerNorm itself is not an identity
    map for arbitrary inputs, so post-norm cannot reach true identity even
    with zeroed sublayer weights. True identity at init is required here —
    see the project's "encoder before kv_cond must start as identity" rule.
    """

    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, attn_dim, attn_num_heads, out_dim, mlp_dim, num_mlp_layers, drop_attn, key):
        attn_key, mlp_key, self.dropout_key = jr.split(key, 3)
        assert attn_dim % attn_num_heads == 0, "attn_dim must be divisible by num_heads"
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=attn_num_heads,
            query_size=attn_dim,
            dropout_p=drop_attn,
            key=attn_key,
        )
        self.mlp = eqx.nn.MLP(
            in_size=attn_dim,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=out_dim,
            key=mlp_key,
        )
        self.norm1 = eqx.nn.LayerNorm(shape=attn_dim)
        self.norm2 = eqx.nn.LayerNorm(shape=attn_dim)

    def __call__(self, x, attn_mask=None):
        h = jax.vmap(self.norm1)(x)
        x = x + self.attn(h, h, h, mask=attn_mask, key=self.dropout_key)
        h = jax.vmap(self.norm2)(x)
        return x + jax.vmap(self.mlp)(h)


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
        agent_proj = eqx.nn.Linear(agent_dim, out_dim, use_bias=False, key=ap)
        # identity init: agent_dim == out_dim always (see scene_encoder), so the
        # combiner's entry token starts as a verbatim copy of agent_encodings —
        # matching ContextCombiner's identity-passthrough behavior at init.
        assert agent_dim == out_dim, (agent_dim, out_dim)
        self.agent_proj = eqx.tree_at(
            lambda l: l.weight, agent_proj, jnp.eye(out_dim, agent_dim)
        )

        def _zero_linear(in_dim, k):
            lin = eqx.nn.Linear(in_dim, out_dim, use_bias=False, key=k)
            return eqx.tree_at(lambda l: l.weight, lin, jnp.zeros_like(lin.weight))

        self.map_proj = _zero_linear(map_dim, mp) if map_dim > 0 else None
        self.tl_proj = _zero_linear(tl_dim, tp) if tl_dim > 0 else None
        # zero-init: added directly onto the agent token (line below), so a
        # random init would corrupt it at step 0 just like agent_proj would.
        self.rel_proj = _zero_linear(rel_dim, rp) if rel_dim > 0 else None

        def _make_zero_init_layer(lk):
            layer = _PreNormSelfAttentionMLP(
                attn_dim=out_dim,
                attn_num_heads=num_heads,
                out_dim=out_dim,
                mlp_dim=hidden_dim,
                num_mlp_layers=2,
                drop_attn=0.0,
                key=lk,
            )
            # zero-init attn out_proj and MLP last layer (incl. biases) →
            # sublayer output is exactly 0 → pre-norm residual is identity
            layer = eqx.tree_at(
                lambda l: l.attn.output_proj.weight, layer, jnp.zeros_like(layer.attn.output_proj.weight)
            )
            last = layer.mlp.layers[-1]
            layer = eqx.tree_at(lambda l: l.mlp.layers[-1].weight, layer, jnp.zeros_like(last.weight))
            if last.bias is not None:
                layer = eqx.tree_at(lambda l: l.mlp.layers[-1].bias, layer, jnp.zeros_like(last.bias))
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
