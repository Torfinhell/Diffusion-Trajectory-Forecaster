import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.diffusion_samplers.ddpm import get_beta_schedule


class DDIMSampler(eqx.Module):
    num_steps: int
    clamp_val: float
    eta: float
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    schedule: str

    def __init__(
        self, steps=100, schedule="cosine", clamp_val=100.0, eta=0.0
    ):
        self.schedule = schedule
        self.clamp_val = clamp_val
        self.eta = eta
        self.num_steps = steps
        betas = get_beta_schedule(variant=self.schedule, num_steps=self.num_steps)
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, 0)

    def add_noise(
        self,
        x_0,  # (F, T)
        noise,  # (F, T)
        timestep,  # (1,)
    ):
        alpha_prod = self.alphas_cumprod[timestep]
        return jnp.sqrt(alpha_prod) * x_0 + jnp.sqrt(1 - alpha_prod) * noise

    def step(
        self,
        key,
        model_output,  # (T, F)
        timestep,  # (1,)
        sample,  # (T, F)
        prediction_type="sample",
    ):
        if prediction_type not in ["sample", "error", "x0", "epsilon"]:
            raise ValueError(f"Invalid prediction_type: {prediction_type}")

        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = jnp.where(
            timestep > 0,
            self.alphas_cumprod[timestep - 1],
            jnp.array(1.0, dtype=self.alphas_cumprod.dtype),
        )

        original_sample = self.pred_original_sample(
            model_output, sample, alpha_prod, prediction_type
        )
        pred_epsilon = self.pred_epsilon(sample, original_sample, alpha_prod)
        sigma = self.ddim_sigma(timestep, alpha_prod, alpha_prod_prev)
        direction = jnp.sqrt(jnp.maximum(1 - alpha_prod_prev - sigma**2, 0.0))
        prev_sample = (
            jnp.sqrt(alpha_prod_prev) * original_sample + direction * pred_epsilon
        )

        noise = jr.normal(key, model_output.shape)
        variance = jnp.where(timestep > 0, sigma * noise, jnp.zeros_like(sample))
        return prev_sample + variance

    def pred_original_sample(
        self,
        model_output,
        sample,
        alpha_prod,
        prediction_type="sample",
    ):
        beta_prod = 1 - alpha_prod
        if prediction_type in ("sample", "x0"):
            original_sample = model_output
        elif prediction_type in ("error", "epsilon"):
            original_sample = (sample - jnp.sqrt(beta_prod) * model_output) / (
                jnp.sqrt(alpha_prod)
            )
        else:
            raise NotImplementedError
        return jnp.clip(original_sample, -self.clamp_val, self.clamp_val)

    def pred_epsilon(self, sample, original_sample, alpha_prod):
        beta_prod = 1 - alpha_prod
        return (sample - jnp.sqrt(alpha_prod) * original_sample) / jnp.sqrt(beta_prod)

    def ddim_sigma(self, timestep, alpha_prod, alpha_prod_prev):
        beta_prod = 1 - alpha_prod
        beta_prod_prev = 1 - alpha_prod_prev
        sigma = self.eta * jnp.sqrt(beta_prod_prev / beta_prod)
        sigma = sigma * jnp.sqrt(1 - alpha_prod / alpha_prod_prev)
        return jnp.where(timestep > 0, sigma, jnp.array(0.0, dtype=sigma.dtype))
