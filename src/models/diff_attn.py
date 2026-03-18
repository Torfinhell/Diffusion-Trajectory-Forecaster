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


class PositionalEncoding(eqx.Module):  # TODO
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc_out: eqx.nn.Linear
    out_shape: tuple[int, ...]

    def __init__(
        self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
    ):
        k1, k2, k3 = jr.split(key, 3)
        traj_dim, cond_dim = prod(output_shape), prod(input_shape)
        self.out_shape = output_shape
        self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
        self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

    def __call__(self, t, x_t, cond):
        x = jnp.concatenate(
            [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
        )
        x = jnn.relu(self.fc1(x))
        x = jnn.relu(self.fc2(x))
        return self.fc_out(x).reshape(self.out_shape)


class SceneEncoder(eqx.Module):
    """
    input_tensor: (B, A, T, F)
    in_dim=B
    out_dim=E
    rnn_full_dim=num_heads*rnn_dim
    """

    rnn_time: eqx.nn.MultiheadAttention | eqx.nn.LSTMCell
    sa_agents: eqx.nn.MultiheadAttention
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
        key,
    ):
        k1, k2, k3, self.dropout_key = jr.split(key, 4)
        self.rnn_type = rnn_type
        in_dim = num_agents * num_feat
        rnn_dim = in_dim
        assert (
            rnn_dim % rnn_num_heads == 0 or rnn_type == "lstm"
        ), "input rnn_dim should be divisable by rnn_num_heads"
        self.rnn_time = (
            eqx.nn.LSTMCell(input_size=in_dim, hidden_size=rnn_dim, key=k1)
            if rnn_type == "lstm"
            else eqx.nn.MultiheadAttention(
                num_heads=rnn_num_heads, query_size=rnn_dim, dropout_p=drop_attn, key=k1
            )
        )
        sa_dim = time_len * num_feat
        assert (
            sa_dim % sa_num_heads == 0
        ), "input sa_dim should be divisable by num_heads"
        self.sa_agents = eqx.nn.MultiheadAttention(
            num_heads=sa_num_heads, query_size=sa_dim, dropout_p=drop_attn, key=k2
        )
        mlp_in_dim = time_len * num_feat
        mlp_out_dim = time_len * num_feat
        self.mlp = eqx.nn.MLP(
            in_size=mlp_in_dim,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=mlp_out_dim,
            key=k3,
        )

    def __call__(self, x):  # (1, A, T, F)
        b, a, t, f = x.shape
        x = rearrange(x, "1 a t f -> t (a f)")  # (T, A*F)
        if isinstance(self.rnn_time, eqx.nn.LSTMCell):

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
        else:
            x = self.rnn_time(x, x, x, key=self.dropout_key)  # mhsa
        x = rearrange(x, "t (a f) -> a (t f)", a=a)
        x = self.sa_agents(x, x, x, key=self.dropout_key)
        return jax.vmap(lambda ag: self.mlp(ag))(x).reshape(1, a, t, f)  # (1, A, T, F)


class SaMLP(eqx.Module):
    scene_encoder: SceneEncoder
    agent_pos_emb: PositionalEncoding
    time_pos_emb: PositionalEncoding
    out_shape: tuple[int, ...]
    sa_mlp: None  # specify here array of self_attention+mlp(multiple blocks)

    def __init__(
        self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
    ):
        k1, k2, k3 = jr.split(key, 3)
        traj_dim, cond_dim = prod(output_shape), prod(input_shape)
        self.out_shape = output_shape
        self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
        self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

    def __call__(self, t, x_t, cond):
        x = jnp.concatenate(
            [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
        )
        x = jnn.relu(self.fc1(x))
        x = jnn.relu(self.fc2(x))
        return self.fc_out(x).reshape(self.out_shape)


class SaCaMLP2(eqx.Module):
    scene_encoder: SceneEncoder
    agent_pos_emb: PositionalEncoding
    time_pos_emb: PositionalEncoding
    out_shape: tuple[int, ...]
    sa_mlp: None  # specify here array of self_attention+mlp(multiple blocks)

    def __init__(
        self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
    ):
        k1, k2, k3 = jr.split(key, 3)
        traj_dim, cond_dim = prod(output_shape), prod(input_shape)
        self.out_shape = output_shape
        self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
        self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

    def __call__(self, t, x_t, cond):
        x = jnp.concatenate(
            [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
        )
        x = jnn.relu(self.fc1(x))
        x = jnn.relu(self.fc2(x))
        return self.fc_out(x).reshape(self.out_shape)


class DiffAttention(eqx.Module):
    scene_encoder: SceneEncoder
    out_shape: tuple[int, ...]
    # sa_ca_mlp_2: list[SaCaMLP2]
    # sa_mlp: list[SaMLP]

    def __init__(
        self,
        se_args,
        sacamlp_args,
        num_saca_mlp,
        sacamlp2_args,
        num_sacamlp2,
        out_shape: list[int],
        key,
    ):
        se_key, sa_ca_mlp_2_key, sacamlp_key = jr.split(key, 3)
        self.scene_encoder = SceneEncoder(**se_args, key=se_key)
        # sacamlp_keys = jr.split(sacamlp_key, num_saca_mlp)
        # self.sa_mlp=[SaMLP(**sacamlp_args, key=sacamlp_key) for _, sacamlp_key in zip(range(num_saca_mlp), sacamlp_keys)]
        # sa_ca_mlp_2_keys = jr.split(sa_ca_mlp_2_key, num_sacamlp2)
        # self.sa_ca_mlp_2=[SaCaMLP2(**sacamlp2_args, key=sa_ca_mlp_2_key) for _, sa_ca_mlp_2_key in zip(range(num_sacamlp2),sa_ca_mlp_2_keys)]
        self.out_shape = out_shape

    def __call__(self, t, x_t, cond):
        KV_cond = self.scene_encoder(cond)
        x = jnp.concat([t, x_t], axis=-1)
        # for layer in self.sa_mlp:
        #     x=layer(x) #TODO apply lax scan
        # for layer in self.sa_ca_mlp_2:
        #     x=layer(x) #TODO apply lax scan
        # x=self.sa_ca_mlp_2(x, KV_cond) #apply lax scan
        return x.reshape(self.out_shape)


class DiffusionAttentionModel(BaseDiffusionModel):
    def __init__(
        self,
        se_args,
        sacamlp_args,
        num_saca_mlp,
        sacamlp2_args,
        num_sacamlp2,
        output_shape: list[int],
        **kwargs
    ):
        self.se_args = se_args
        self.sacamlp_args = sacamlp_args
        self.num_saca_mlp = num_saca_mlp
        self.sacamlp2_args = sacamlp2_args
        self.num_sacamlp2 = num_sacamlp2
        self.output_shape = output_shape
        super().__init__(**kwargs)

    def get_model(self, key_model):
        return DiffAttention(
            se_args=self.se_args,
            sacamlp_args=self.sacamlp_args,
            num_saca_mlp=self.num_saca_mlp,
            sacamlp2_args=self.sacamlp2_args,
            num_sacamlp2=self.num_sacamlp2,
            out_shape=self.output_shape,
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
