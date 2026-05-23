import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class LSTMTemporal(eqx.Module):
    cell: eqx.nn.LSTMCell

    def __init__(self, in_dim: int, hidden_dim: int, key):
        self.cell = eqx.nn.LSTMCell(input_size=in_dim, hidden_size=hidden_dim, key=key)

    def __call__(self, x, pos_emb):
        del pos_emb

        def scan_fn(state, xt):
            new_state = self.cell(xt, state)
            return new_state, new_state[0]

        init_state = (
            jnp.zeros((self.cell.hidden_size,)),
            jnp.zeros((self.cell.hidden_size,)),
        )
        _, x = jax.lax.scan(scan_fn, init_state, x)
        return x


class MHSATemporal(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey

    def __init__(self, dim: int, num_heads: int, drop_attn: float, key):
        attn_key, self.dropout_key = jr.split(key, 2)
        assert dim % num_heads == 0, "temporal dim must be divisible by num_heads"
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=dim,
            dropout_p=drop_attn,
            key=attn_key,
        )

    def __call__(self, x, pos_emb):
        return self.attn(
            x,
            x,
            x,
            key=self.dropout_key,
            process_heads=pos_emb.process_heads,
        )


_TEMPORAL = {
    "lstm": LSTMTemporal,
    "mhsa": MHSATemporal,
}


def build_temporal(name: str, *, in_dim: int, num_heads: int, drop_attn: float, key):
    key_name = name.strip().lower()
    if key_name not in _TEMPORAL:
        raise ValueError(f"Unknown rnn_type {name!r}; choose from {sorted(_TEMPORAL)}")
    cls = _TEMPORAL[key_name]
    if cls is LSTMTemporal:
        return cls(in_dim=in_dim, hidden_dim=in_dim, key=key)
    return cls(dim=in_dim, num_heads=num_heads, drop_attn=drop_attn, key=key)
