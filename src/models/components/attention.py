import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class SelfAttentionMLP(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        attn_dim: int,
        attn_num_heads: int,
        out_dim: int,
        mlp_dim: int,
        num_mlp_layers: int,
        drop_attn: float,
        key,
    ):
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
        x = jax.vmap(self.norm1)(
            x + self.attn(x, x, x, mask=attn_mask, key=self.dropout_key)
        )
        return jax.vmap(self.norm2)(x + jax.vmap(self.mlp)(x))


class CrossAttentionMLP(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        attn_dim: int,
        attn_num_heads: int,
        out_dim: int,
        mlp_dim: int,
        num_mlp_layers: int,
        drop_attn: float,
        kv_dim: int,
        key,
    ):
        attn_key, mlp_key, self.dropout_key = jr.split(key, 3)
        assert attn_dim % attn_num_heads == 0, "attn_dim must be divisible by num_heads"
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=attn_num_heads,
            query_size=attn_dim,
            key_size=kv_dim,
            value_size=kv_dim,
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

    def __call__(self, x, kv_cond, attn_mask=None):
        x = jax.vmap(self.norm1)(
            x + self.attn(x, kv_cond, kv_cond, mask=attn_mask, key=self.dropout_key)
        )
        return jax.vmap(self.norm2)(x + jax.vmap(self.mlp)(x))


class TransformerEncoder(eqx.Module):
    layers: list[SelfAttentionMLP]

    def __init__(
        self,
        layers: int,
        attn_dim: int,
        attn_num_heads: int,
        mlp_dim: int,
        num_mlp_layers: int,
        drop_attn: float,
        key,
    ):
        layer_keys = jr.split(key, layers)
        self.layers = [
            SelfAttentionMLP(
                attn_dim=attn_dim,
                attn_num_heads=attn_num_heads,
                out_dim=attn_dim,
                mlp_dim=mlp_dim,
                num_mlp_layers=num_mlp_layers,
                drop_attn=drop_attn,
                key=layer_key,
            )
            for layer_key in layer_keys
        ]

    def __call__(self, context_tokens, context_mask):
        valid_context = ~context_mask
        self_attn_mask = valid_context[:, None] & valid_context[None, :]
        tokens = jnp.where(context_mask[:, None], 0.0, context_tokens)
        for layer in self.layers:
            tokens = layer(tokens, attn_mask=self_attn_mask)
            tokens = jnp.where(context_mask[:, None], 0.0, tokens)
        return tokens
