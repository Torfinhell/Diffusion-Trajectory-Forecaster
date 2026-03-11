# from math import prod

# import equinox as eqx
# import jax.nn as jnn
# import jax.numpy as jnp
# import jax.random as jr
# import optax

# from .base_model import BaseDiffusionModel
# class PositionalEncoding(eqx.Module): #TODO
#     fc1: eqx.nn.Linear
#     fc2: eqx.nn.Linear
#     fc_out: eqx.nn.Linear
#     out_shape: tuple[int, ...]

#     def __init__(
#         self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
#     ):
#         k1, k2, k3 = jr.split(key, 3)
#         traj_dim, cond_dim = prod(output_shape), prod(input_shape)
#         self.out_shape = output_shape
#         self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
#         self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
#         self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

#     def __call__(self, t, x_t, cond):
#         x = jnp.concatenate(
#             [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
#         )
#         x = jnn.relu(self.fc1(x))
#         x = jnn.relu(self.fc2(x))
#         return self.fc_out(x).reshape(self.out_shape)
# class SceneEncoder(eqx.Module):
#     rnn_time: eqx.nn.MultiheadAttention | eqx.nn.LSTMCell
#     sa_agents: eqx.nn.MultiheadAttention
#     mlp_layers: list[eqx.nn.MLP]

#     def __init__(
#         self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
#     ):
#         k1, k2, k3 = jr.split(key, 3)
#         traj_dim, cond_dim = prod(output_shape), prod(input_shape)
#         self.out_shape = output_shape
#         self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
#         self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
#         self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

#     def __call__(self, ):
#         x = jnp.concatenate(
#             [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
#         )
#         x = jnn.relu(self.fc1(x))
#         x = jnn.relu(self.fc2(x))
#         return self.fc_out(x).reshape(self.out_shape)
# class SaMLP(eqx.Module):
#     scene_encoder: SceneEncoder
#     agent_pos_emb: PositionalEncoding
#     time_pos_emb: PositionalEncoding
#     out_shape: tuple[int, ...]
#     sa_mlp: #specify here array of self_attention+mlp(multiple blocks)

#     def __init__(
#         self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
#     ):
#         k1, k2, k3 = jr.split(key, 3)
#         traj_dim, cond_dim = prod(output_shape), prod(input_shape)
#         self.out_shape = output_shape
#         self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
#         self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
#         self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

#     def __call__(self, t, x_t, cond):
#         x = jnp.concatenate(
#             [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
#         )
#         x = jnn.relu(self.fc1(x))
#         x = jnn.relu(self.fc2(x))
#         return self.fc_out(x).reshape(self.out_shape)
# class SaCaMLP2(eqx.Module):
#     scene_encoder: SceneEncoder
#     agent_pos_emb: PositionalEncoding
#     time_pos_emb: PositionalEncoding
#     out_shape: tuple[int, ...]
#     sa_mlp: #specify here array of self_attention+mlp(multiple blocks)

#     def __init__(
#         self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
#     ):
#         k1, k2, k3 = jr.split(key, 3)
#         traj_dim, cond_dim = prod(output_shape), prod(input_shape)
#         self.out_shape = output_shape
#         self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
#         self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
#         self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

#     def __call__(self, t, x_t, cond):
#         x = jnp.concatenate(
#             [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
#         )
#         x = jnn.relu(self.fc1(x))
#         x = jnn.relu(self.fc2(x))
#         return self.fc_out(x).reshape(self.out_shape)
# class DiffAttention(eqx.Module):
#     scene_encoder: SceneEncoder
#     agent_pos_emb: PositionalEncoding
#     time_pos_emb: PositionalEncoding
#     out_shape: tuple[int, ...]
#     sa_ca_mlp_2: list[SaCaMLP2]
#     sa_mlp: list[SaMLP]

#     def __init__(
#         self, hid_dim: int, input_shape: list[int], output_shape: list[int], key
#     ):
#         k1, k2, k3 = jr.split(key, 3)
#         self.scene_encoder=SceneEncoder()
#         # traj_dim, cond_dim = prod(output_shape), prod(input_shape)
#         # self.out_shape = output_shape
#         # self.fc1 = eqx.nn.Linear(traj_dim + cond_dim + 1, hid_dim, key=k1)
#         # self.fc2 = eqx.nn.Linear(hid_dim, hid_dim, key=k2)
#         # self.fc_out = eqx.nn.Linear(hid_dim, traj_dim, key=k3)

#     def __call__(self, t, x_t, cond):
#         x = jnp.concatenate(
#             [x_t.reshape(-1), cond.reshape(-1), jnp.atleast_1d(t)], axis=0
#         )
#         x = jnn.relu(self.fc1(x))
#         x = jnn.relu(self.fc2(x))
#         return self.fc_out(x).reshape(self.out_shape)


# class DiffusionAttentionModel(BaseDiffusionModel):
#     def __init__(
#         self, hid_dim, input_shape: list[int], output_shape: list[int], **kwargs
#     ):
#         self.hid_dim = hid_dim
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         super().__init__(**kwargs)

#     def get_model(self, key_model):
#         return DiffAttention(
#             hid_dim=self.hid_dim,
#             input_shape=self.input_shape,
#             output_shape=self.output_shape,
#             key=key_model,
#         )

#     def configure_optimizers(self):
#         self.optim = optax.adam(3e-4)
#         self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

#     def configure_ddpm_scheduler(self):
#         self.int_beta = lambda t: t
#         self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
#         self.t1 = 10.0
#         self.dt0 = 0.1
