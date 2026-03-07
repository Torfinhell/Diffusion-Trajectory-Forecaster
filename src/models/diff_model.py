from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from hydra.utils import instantiate

from .base_model import BaseDiffusionModel


class DiffusionModel(BaseDiffusionModel):
    def __init__(self,**kwargs,):
        super().__init__(**kwargs,)

    
    def get_data(self, batch, t):
        traj = batch.log_trajectory

        # past trajectory (context)
        past_xy = jnp.stack(
            [traj.x[..., :t], traj.y[..., :t]],
            axis=-1,
        )  # [B, N, T_hist, 2]

        # future trajectory (target for diffusion)
        future_xy = jnp.stack(
            [traj.x[..., t:],
            traj.y[..., t:]],
            axis=-1,
        )  # [B, N, H, 2]

        # mask for valid objects
        valid = traj.valid[..., t:]  # [B, N, H]

        return {
            "traj": future_xy,
            "context": past_xy,
            "mask": valid,
        }



    def get_model(self):
        future_steps = self.cfg_model.args.traj_len - self.history
        return instantiate(
            self.cfg_model.model,
            out_dim=2 * future_steps,
            hid_dim=self.cfg_model.args.hid_dim,
            history=self.history,
            key=self.model_key,
        )

    def configure_optimizers(self):
        self.optim = optax.adam(3e-4)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

    @staticmethod
    def single_loss_fn(model, weight, int_beta, data, t, key):
        """
        OU process provides analytical mean and variance
        int_beta(t) = ß = θ
        E[X_t] = μ + exp[-θ t] ( X_0 - μ) w/ μ=0 gives =X_0 * exp[ - θ t ]
        V[X_t] = σ^2/(2θ) ( 1 - exp(-2 θ t) ) w/ σ^2=ß=θ gives = 1 - exp(-2 ß t)
        :param model:
        :param weight:
        :param int_beta:
        :param data:
        :param t:
        :param key:
        :return:
        """
        traj = data["traj"]
        context = data["context"]
        mask = data.get("mask", None)

        N = traj.shape[0]

        # flatten per agent
        traj = traj.reshape(N, -1)         # [N, H*2]
        context = context.reshape(N, -1)   # [N, T*2]

        mean = traj * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)

        key_agents = jr.split(key, N)
        noise = jax.vmap(lambda k: jr.normal(k, (traj.shape[1],)))(key_agents)   # [N, H*2]
        y = mean + std * noise                                                    # [N, H*2]

        # shared model applied per agent
        pred = jax.vmap(lambda y_i, c_i: model(t, y_i, c_i))(y, context)          # [N, H*2]

        err = (pred + noise / std) ** 2

        if mask is not None:
            mask = jnp.repeat(mask[..., None], 2, axis=-1)   # [N, H, 2]
            mask = mask.reshape(N, -1).astype(err.dtype)     # [N, H*2]
            loss = (err * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        else:
            loss = jnp.mean(err)

        return weight(t) * loss

    @staticmethod
    def batch_loss_fn(model, weight, int_beta, data, t1, key):
        batch_size = data["traj"].shape[0]
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        """
		Low-discrepancy sampling over t to reduce variance
		by sampling very evenly by sampling uniformly and independently from (t1-t0)/batch_size bins
		t = [U(0,1), U(1,2), U(2,3), ...]
		"""
        t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
        t = t + (t1 / batch_size) * jnp.arange(batch_size)
        """ Fixing the first three arguments of single_loss_fn, leaving data, t and key as input """
        loss_fn = partial(DiffusionModel.single_loss_fn, model, weight, int_beta)
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(data, t, losskey))
