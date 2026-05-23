import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class NullTimePosEmb(eqx.Module):
    def process_heads(self, query_heads, key_heads, value_heads):
        return query_heads, key_heads, value_heads

    def after_temporal(self, x, time_len: int):
        del time_len
        return x


class LookupTimePosEmb(eqx.Module):
    embed: eqx.nn.Embedding

    def __init__(self, time_len: int, embedding_size: int, key):
        self.embed = eqx.nn.Embedding(
            num_embeddings=time_len,
            embedding_size=embedding_size,
            key=key,
        )

    def process_heads(self, query_heads, key_heads, value_heads):
        return query_heads, key_heads, value_heads

    def after_temporal(self, x, time_len: int):
        return x + self.embed(jnp.arange(time_len))


class RoPETimePosEmb(eqx.Module):
    rope: eqx.nn.RotaryPositionalEmbedding

    def __init__(self, embedding_size: int, rope_theta: float, key):
        self.rope = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=embedding_size,
            theta=rope_theta,
            key=key,
        )

    def process_heads(self, query_heads, key_heads, value_heads):
        query_heads = jax.vmap(self.rope, in_axes=1, out_axes=1)(query_heads)
        key_heads = jax.vmap(self.rope, in_axes=1, out_axes=1)(key_heads)
        return query_heads, key_heads, value_heads

    def after_temporal(self, x, time_len: int):
        del time_len
        return x


_TIME_POS_EMB = {
    "none": NullTimePosEmb,
    "lookup": LookupTimePosEmb,
    "rope": RoPETimePosEmb,
}


def build_time_pos_emb(
    name: str, *, time_len: int, embedding_size: int, rope_theta: float, key
):
    emb_name = name.strip().lower()
    if emb_name not in _TIME_POS_EMB:
        raise ValueError(
            f"Unknown pos_emb_type {name!r}; choose from {sorted(_TIME_POS_EMB)}"
        )
    cls = _TIME_POS_EMB[emb_name]
    if cls is NullTimePosEmb:
        return cls()
    if cls is RoPETimePosEmb:
        return cls(embedding_size=embedding_size, rope_theta=rope_theta, key=key)
    return cls(time_len=time_len, embedding_size=embedding_size, key=key)
