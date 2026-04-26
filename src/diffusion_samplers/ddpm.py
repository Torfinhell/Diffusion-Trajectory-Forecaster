import equinox as eqx
import jax
import jax.numpy as jnp


def get_beta_schedule(self, variant, num_steps):
    if variant == "cosine":
        # sigma = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-t), 1e-5))
        pass
    else:
        raise NotImplementedError("Beta Schedule is not defined")


class DDPMSampler(eqx.Module):
    num_steps: int
    clamp_val: float
    betas: jax.Array
    alphas: jax.Array

    def __init__(
        self, steps=50, schedule="cosine", clamp_val=100.0
    ):  # TODO fix clamp_val
        self.num_steps = steps
        self.schedule = schedule
        self.clamp_val = clamp_val
        betas = self.get_beta_schedule(variant=schedule, num_steps=self.num_steps)
        self.betas = jnp.sqrt(betas)
        self.alphas = 1 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, 0)  # TODO fix the [1:]

    def add_noise(  # TODO No grad
        self,
        x_0,  # (B, F, T) (F, T) Tensor at different denoise levels defined by timestemp
        noise,  # (B, F, T)
        timesteps,  # (B,)
    ):
        coeffs_2 = self.alphas_cumprod[timesteps]
        return jnp.sqrt(coeffs_2) * x_0 + jnp.sqrt(1 - coeffs_2) * noise

    def get_noise(self):
        pass

    def step(self):
        pass

    def q_mean(self):
        pass

    def q_x0(self):
        pass

    def q_variance(self):
        pass
