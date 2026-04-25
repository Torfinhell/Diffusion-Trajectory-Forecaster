from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange, repeat

from utils.stats import masked_abs_mean

from .base_model import BaseDiffusionModel
from .base_model_debug import DebuggableBaseDiffusionModel


class FourierEmbedding(eqx.Module):
    # mlp_out:eqx.nn.Linear
    freqs: eqx.nn.Embedding
    embed_dim: int

    def __init__(self, embed_dim, key):
        self.freqs = eqx.nn.Embedding(1, embed_dim // 2, key=key)
        self.embed_dim = embed_dim

    def __call__(self, x):  # x is scalar -> (embed_dim,)
        return jnp.concat(
            [jnp.cos(self.freqs.weight * x), jnp.sin(self.freqs.weight * x)], axis=-1
        ).squeeze(0)[: self.embed_dim]


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
        if x.ndim == 3:
            x = x[None, ...]
        elif x.ndim != 4:
            raise ValueError(f"SceneEncoder expected 3D or 4D input, got {x.shape}")
        b, a, t, f = x.shape
        expected_in_dim = (
            self.rnn_time.input_size
            if isinstance(self.rnn_time, eqx.nn.LSTMCell)
            else self.rnn_time.query_size
        )
        actual_in_dim = a * f
        if actual_in_dim != expected_in_dim:
            raise ValueError(
                "SceneEncoder input shape mismatch: expected A*F="
                f"{expected_in_dim} from model config, got A*F={actual_in_dim} "
                f"(A={a}, F={f}). Align model.num_agents/num_feat with dataset preprocessing."
            )
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
    embed_future: FourierEmbedding
    embed_past: FourierEmbedding
    mlp_out: eqx.nn.Linear

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
        se_key, ca_mlp_key, sacamlp_key, out_key, fourier_key_1, fourier_key_2 = (
            jr.split(key, 6)
        )
        self.scene_encoder = SceneEncoder(**se_args, key=se_key)
        sacamlp_keys = jr.split(sacamlp_key, num_sa_mlp)
        self.sa_mlp_layers = [
            AttentionMLP(**samlp_args, key=sacamlp_key, type_attn="self")
            for _, sacamlp_key in zip(range(num_sa_mlp), sacamlp_keys)
        ]
        ca_mlp_keys = jr.split(ca_mlp_key, num_camlp)
        self.ca_mlp_layers = [
            AttentionMLP(**camlp_args, key=ca_mlp_key, type_attn="cross")
            for _, ca_mlp_key in zip(range(num_camlp), ca_mlp_keys[:-1])
        ]
        self.embed_future = FourierEmbedding(camlp_args["out_dim"], key=fourier_key_1)
        self.embed_past = FourierEmbedding(se_args["out_dim"], key=fourier_key_2)
        self.mlp_out = eqx.nn.Linear(
            in_features=camlp_args["out_dim"], out_features=final_out_dim, key=out_key
        )
        self.out_shape = out_shape

    def __call__(self, t_noise, x_t, cond):
        if x_t.ndim == 3:
            x_t = x_t[None, ...]
        elif x_t.ndim != 4:
            raise ValueError(
                f"DiffAttention expected x_t with 3 or 4 dims, got {x_t.shape}"
            )
        KV_cond = self.scene_encoder(cond)  # (A, out_dim)
        _, a, t, f = x_t.shape
        x_t = x_t.reshape(a, -1) + self.embed_future(t_noise)  # (A, dim) + (dim,)
        KV_cond = KV_cond + self.embed_past(t_noise)  # (A, out_dim) + (out_dim,)
        for layer in self.sa_mlp_layers:
            x_t = layer(x_t)
        for layer in self.ca_mlp_layers:
            x_t = layer(x_t, KV_cond)
        return jax.vmap(self.mlp_out)(x_t).reshape(self.out_shape)


class _DiffusionAttentionBase:
    def __init__(
        self,
        se_args,
        samlp_args,
        num_sa_mlp,
        camlp_args,
        num_camlp,
        final_out_dim,
        output_shape: list[int],
        lr=4e-4,
        lr_scheduler=None,
        **kwargs,
    ):
        self.se_args = se_args
        self.samlp_args = samlp_args
        self.num_sa_mlp = num_sa_mlp
        self.camlp_args = camlp_args
        self.num_camlp = num_camlp
        self.output_shape = output_shape
        self.final_out_dim = final_out_dim
        self.lr = lr
        self.lr_scheduler_cfg = lr_scheduler or {"name": "none"}
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
        self.optim = self.build_optimizer(self.build_learning_rate(self.lr))
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

    def configure_ddpm_scheduler(self):
        self.int_beta = lambda t: t
        self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
        self.t0 = 1e-3
        self.t1 = 2.0
        self.dt0 = 0.01

    def single_loss_and_stats_fn(model, int_beta, batch, t, key):
        """
        OU process provides analytical mean and variance
        int_beta(t) = ß = θ
        E[X_t] = μ + exp[-θ t] ( X_0 - μ) w/ μ=0 gives =X_0 * exp[ - θ t ]
        V[X_t] = σ^2/(2θ) ( 1 - exp(-2 θ t) ) w/ σ^2=ß=θ gives = 1 - exp(-2 ß t)
        :param model:
        :param weight:
        :param int_beta:
        :param batch:
        :param t:
        :param key:
        :return:
        """
        gt_xy = batch["agent_future"][..., :2]
        INPUT_DIM = gt_xy.shape
        mean = gt_xy * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, INPUT_DIM)
        y = mean + std * noise
        pred_xy = model(t, y, batch["agent_past"])
        err = (pred_xy - gt_xy) ** 2
        weights = jnp.asarray(batch["agents_coeffs"], dtype=err.dtype)[
            ..., None, None
        ] * jnp.asarray(batch["agent_future_valid"], dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        weighted_element_count = jnp.ones_like(err) * weights
        loss = (err * weights).sum() / jnp.maximum(weighted_element_count.sum(), 1.0)
        valid_weights = jnp.asarray(batch["agent_future_valid"], dtype=gt_xy.dtype)
        stats = {
            "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
            "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
            "valid_ratio": jnp.mean(valid_weights),
        }
        return loss, stats


class DiffusionAttentionModel(_DiffusionAttentionBase, BaseDiffusionModel):
    pass


class DiffusionAttentionDebugModel(
    _DiffusionAttentionBase, DebuggableBaseDiffusionModel
):
    pass
