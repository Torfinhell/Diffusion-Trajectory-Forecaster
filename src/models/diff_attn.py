from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange, repeat

from .base_model import BaseDiffusionModel


class SceneEncoder(eqx.Module):
    """
    input_tensor: (B, A, T, F)
    in_dim=B
    out_dim=E
    rnn_full_dim=num_heads*rnn_dim
    """

    pos_emb_type: Literal["rope", "Trainable", "None"]  # for time
    rnn_time: eqx.nn.MultiheadAttention | eqx.nn.LSTMCell
    sa_agents: eqx.nn.MultiheadAttention
    embedding: eqx.nn.Embedding | eqx.nn.RotaryPositionalEmbedding
    mlp: eqx.nn.MLP
    rnn_type: str
    dropout_key: jax.random.PRNGKey

    def __init__(
        self,
        rnn_type: Literal["lstm", "mhsa"],  # one of "lstm" "mhsa"
        rnn_num_heads: int,
        sa_num_heads: int,
        drop_attn: float,
        mlp_dim: int,
        num_mlp_layers: int,
        num_agents: int,
        time_len: int,
        num_feat: int,
        out_dim: int,
        pos_emb_type: Literal["rope", "lookup", "None"],
        rope_theta: float,
        key,
    ):
        rnn_key, sa_key, mlp_key, self.dropout_key, embed_key = jr.split(key, 5)
        self.rnn_type = rnn_type
        in_dim = num_agents * num_feat
        rnn_dim = in_dim
        assert (
            rnn_dim % rnn_num_heads == 0 or rnn_type == "lstm"
        ), "input rnn_dim should be divisable by rnn_num_heads"
        self.rnn_time = (
            eqx.nn.LSTMCell(input_size=in_dim, hidden_size=rnn_dim, key=rnn_key)
            if rnn_type == "lstm"
            else eqx.nn.MultiheadAttention(
                num_heads=rnn_num_heads,
                query_size=rnn_dim,
                dropout_p=drop_attn,
                key=rnn_key,
            )
        )
        sa_dim = time_len * num_feat
        assert (
            sa_dim % sa_num_heads == 0
        ), "input sa_dim should be divisable by num_heads"
        self.sa_agents = eqx.nn.MultiheadAttention(
            num_heads=sa_num_heads, query_size=sa_dim, dropout_p=drop_attn, key=sa_key
        )
        mlp_in_dim = time_len * num_feat
        self.mlp = eqx.nn.MLP(
            in_size=mlp_in_dim,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=out_dim,
            key=mlp_key,
        )
        self.pos_emb_type = pos_emb_type
        embedding_size = num_agents * num_feat
        if self.pos_emb_type == "rope":
            self.embedding = eqx.nn.RotaryPositionalEmbedding(
                embedding_size=embedding_size, theta=rope_theta
            )
        elif self.pos_emb_type == "lookup":
            self.embedding = eqx.nn.Embedding(
                num_embeddings=time_len, embedding_size=embedding_size, key=embed_key
            )
        else:
            self.embedding = None

    def __call__(self, x):  # (1, A, T, F)
        b, a, t, f = x.shape
        x = rearrange(x, "1 a t f -> t (a f)")  # (T, A*F)
        if isinstance(self.rnn_time, eqx.nn.LSTMCell):
            assert not isinstance(self.embedding, eqx.nn.RotaryPositionalEmbedding)

            def scan_fn(state, xt):
                new_state = self.rnn_time(
                    xt, state
                )  # xt: (A*F), state: tuple of (hidden)
                return new_state, new_state[0]  # output hidden states

            init_state = (
                jnp.zeros((self.rnn_time.hidden_size)),
                jnp.zeros((self.rnn_time.hidden_size)),
            )
            _, x = jax.lax.scan(scan_fn, init_state, x)  # x: (T, A*F)
            if isinstance(self.embedding, eqx.nn.Embedding):
                x += self.embedding(jnp.arange(0, t))
        else:
            if isinstance(self.embedding, eqx.nn.RotaryPositionalEmbedding):

                def process_heads(query_heads, key_heads, value_heads):
                    query_heads = jax.vmap(self.embedding, in_axes=1, out_axes=1)(
                        query_heads
                    )
                    key_heads = jax.vmap(self.embedding, in_axes=1, out_axes=1)(
                        key_heads
                    )

                    return query_heads, key_heads, value_heads

            else:
                process_heads = None
            x = self.rnn_time(
                x, x, x, key=self.dropout_key, process_heads=process_heads
            )  # mhsa
            if isinstance(self.embedding, eqx.nn.Embedding):
                x += jax.vmap(self.embedding)(jnp.arange(0, t))

        x = rearrange(x, "t (a f) -> a (t f)", a=a)
        x = self.sa_agents(x, x, x, key=self.dropout_key)
        return jax.vmap(lambda ag: self.mlp(ag))(x).reshape(a, -1)  # (1, A, out_dim)


class AttentionMLP(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey
    type_attn: Literal["cross", "self"]
    mlp: eqx.nn.MLP

    def __init__(
        self,
        attn_dim: int,
        attn_num_heads: int,
        out_dim: int,
        mlp_dim: int,
        num_mlp_layers: int,
        drop_attn: float,
        type_attn: Literal["cross", "self"],
        key,
        kv_dim=None,
    ):
        attn_key, mlp_key, self.dropout_key = jr.split(key, 3)
        assert (
            attn_dim % attn_num_heads == 0
        ), "input attn_dim should be diviattnble by num_heads"
        self.type_attn = type_attn
        if self.type_attn == "self":
            self.attn = eqx.nn.MultiheadAttention(
                num_heads=attn_num_heads,
                query_size=attn_dim,
                dropout_p=drop_attn,
                key=attn_key,
            )
        elif self.type_attn == "cross":
            self.attn = eqx.nn.MultiheadAttention(
                num_heads=attn_num_heads,
                query_size=attn_dim,
                key_size=kv_dim,
                value_size=kv_dim,
                dropout_p=drop_attn,
                key=attn_key,
            )
        mlp_in_dim = attn_dim
        self.mlp = eqx.nn.MLP(
            in_size=mlp_in_dim,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=out_dim,
            key=mlp_key,
        )

    def __call__(self, x, kv_cond=None):  # x: (1, a, t_future*f)
        a, _ = x.shape
        if self.type_attn == "self":
            x = self.attn(x, x, x, key=self.dropout_key)
        elif self.type_attn == "cross":
            x = self.attn(
                x, kv_cond, kv_cond, mask=jnp.diag(jnp.ones(a)), key=self.dropout_key
            )
        return jax.vmap(self.mlp)(x)


class DiffAttention(eqx.Module):
    scene_encoder: SceneEncoder
    out_shape: tuple[int, ...]
    ca_mlp_layers: list[AttentionMLP]
    sa_mlp_layers: list[AttentionMLP]
    num_sa_mlp: int

    def __init__(
        self,
        se_args,
        samlp_args,
        num_sa_mlp,
        camlp_args,
        num_camlp,
        out_shape: list[int],
        final_out_dim: int,
        key,
    ):
        se_key, ca_mlp_key, sacamlp_key = jr.split(key, 3)
        self.scene_encoder = SceneEncoder(**se_args, key=se_key)
        self.num_sa_mlp = num_sa_mlp
        sacamlp_keys = jr.split(sacamlp_key, num_sa_mlp)
        self.sa_mlp_layers = [
            AttentionMLP(**samlp_args, key=sacamlp_key, type_attn="self")
            for _, sacamlp_key in zip(range(num_sa_mlp), sacamlp_keys)
        ]
        ca_mlp_keys = jr.split(ca_mlp_key, num_camlp)
        self.ca_mlp_layers = [
            AttentionMLP(**camlp_args, key=ca_mlp_key, type_attn="cross")
            for _, ca_mlp_key in zip(range(num_camlp - 1), ca_mlp_keys[:-1])
        ]
        if num_camlp > 1:
            camlp_args["out_dim"] = final_out_dim
            self.ca_mlp_layers.append(
                AttentionMLP(**camlp_args, key=ca_mlp_key, type_attn="cross")
            )
        self.out_shape = out_shape

    def __call__(self, t_noise, x_t, cond):
        KV_cond = self.scene_encoder(cond)  # (1, A, out_dim)
        _, a, t, f = x_t.shape
        x_t = jnp.concat(
            [repeat(t_noise, " -> 1 a 1 f", a=a, f=f), x_t], axis=-2
        ).reshape(a, -1)
        # x_t, _=jax.lax.scan(lambda input, layer_idx:(self.sa_mlp_layers[layer_idx](input), None), x_t.reshape(a, -1), jnp.arange(self.num_sa_mlp)) #TODO
        for layer in self.sa_mlp_layers:
            x_t = layer(x_t)
        for layer in self.ca_mlp_layers:
            x_t = layer(x_t, KV_cond)
        return x_t.reshape(self.out_shape)


class DiffusionAttentionModel(BaseDiffusionModel):
    def __init__(
        self,
        se_args,
        samlp_args,
        num_sa_mlp,
        camlp_args,
        num_camlp,
        final_out_dim,
        output_shape: list[int],
        **kwargs
    ):
        self.se_args = se_args
        self.samlp_args = samlp_args
        self.num_sa_mlp = num_sa_mlp
        self.camlp_args = camlp_args
        self.num_camlp = num_camlp
        self.output_shape = output_shape
        self.final_out_dim = final_out_dim
        super().__init__(**kwargs)

    def get_model(self, key_model):
        return DiffAttention(
            se_args=self.se_args,
            samlp_args=self.samlp_args,
            num_sa_mlp=self.num_sa_mlp,
            camlp_args=self.camlp_args,
            num_camlp=self.num_camlp,
            out_shape=self.output_shape,
            final_out_dim=self.final_out_dim,
            key=key_model,
        )

    def configure_optimizers(self):
        self.optim = optax.adam(3e-4)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

    def configure_ddpm_scheduler(self):
        self.int_beta = lambda t: t
        self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
        self.t1 = 10.0
        self.dt0 = 0.1
