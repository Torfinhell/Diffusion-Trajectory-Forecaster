from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .base_model import BaseDiffusionModel


class DiffusionModel(BaseDiffusionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        """
        Get FCOS model from torchvision configured for bounding box predictions.
        Returns a FCOS ResNet50 FPN model with pretrained backbone for custom number of classes.
        """

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
        mean = data * jnp.exp(-0.5 * int_beta(t))  # analytical mean of OU process
        var = jnp.maximum(
            1 - jnp.exp(-int_beta(t)), 1e-5
        )  # analytical variance of OU process
        std = jnp.sqrt(var)
        noise = jr.normal(key, data.shape)
        y = mean + std * noise
        pred = model(t, y)
        return weight(t) * jnp.mean((pred + noise / std) ** 2)  # loss

    @staticmethod
    def batch_loss_fn(model, weight, int_beta, data, t1, key):
        batch_size = data.shape[0]
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
        loss_fn = partial(BaseDiffusionModel.single_loss_fn, model, weight, int_beta)
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(data, t, losskey))
