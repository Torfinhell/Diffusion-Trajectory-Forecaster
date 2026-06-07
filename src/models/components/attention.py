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


class GatedCrossAttentionMLP(eqx.Module):
    """Pre-norm cross-attention with a learned per-layer gate, identity at init.

    Used for *full* cross-attention (each agent query attends to all context
    tokens). Full CA is a documented failure mode here because the denoiser
    sees arbitrary, randomly-mixed kv_cond from step 1 and cannot learn x0.

    This block starts as an exact identity (gate g=0 -> tanh(0)=0 ->
    x + 0*sublayer = x), mirroring the identity-init invariant that makes
    diagonal CA work. As training proceeds the gates open and context
    attention ramps in smoothly instead of corrupting the signal up front.
    Pre-norm (not post-norm) is required so the zero-gate path is truly x.
    """

    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    gate_attn: jax.Array
    gate_mlp: jax.Array

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
        assert attn_dim == out_dim, "gated residual requires attn_dim == out_dim"
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
        # zero-init gates -> tanh(0)=0 -> each sublayer contributes nothing at
        # init -> block is identity. Learned, so context ramps in over training.
        self.gate_attn = jnp.zeros(())
        self.gate_mlp = jnp.zeros(())

    def __call__(self, x, kv_cond, attn_mask=None):
        h = jax.vmap(self.norm1)(x)
        x = x + jnp.tanh(self.gate_attn) * self.attn(
            h, kv_cond, kv_cond, mask=attn_mask, key=self.dropout_key
        )
        h = jax.vmap(self.norm2)(x)
        return x + jnp.tanh(self.gate_mlp) * jax.vmap(self.mlp)(h)


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
