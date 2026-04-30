import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def get_beta_schedule(variant, num_steps):
    if variant == "cosine":
        # alpha_bar(t) = cos(((t/T + s) / (1 + s)) * pi/2)^2
        # betas_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        timesteps = jnp.linspace(0.0, float(num_steps), num_steps + 1)
        alpha_bar = (
            jnp.cos(
                ((timesteps / float(num_steps) + 0.008) / (1.0 + 0.008))
                * (jnp.pi / 2.0)
            )
            ** 2
        )
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return jnp.clip(betas, 1e-8, 0.999)
    else:
        raise NotImplementedError("Beta Schedule is not defined")


class DDPMSampler(eqx.Module):
    num_steps: int
    clamp_val: float
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    schedule: str

    def __init__(
        self, steps=50, schedule="cosine", clamp_val=100.0
    ):  # TODO fix clamp_val
        self.schedule = schedule
        self.clamp_val = clamp_val
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
        coeffs_2 = self.alphas_cumprod[timestep]
        return jnp.sqrt(coeffs_2) * x_0 + jnp.sqrt(1 - coeffs_2) * noise

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
        pred_prev_mean = self.q_mean(model_output, timestep, sample, prediction_type)
        noise = jr.normal(key, model_output.shape)
        variance = jnp.where(
            timestep > 0,
            (self.q_variance(timestep) ** 0.5) * noise,
            jnp.zeros_like(sample),
        )
        return pred_prev_mean + variance

    def q_mean(
        self,
        model_output,  # (T, F)
        timestep,  # (1,)
        sample,  # (T, F)
        prediction_type="sample",
    ):
        alpha_prod = self.alphas_cumprod[timestep]
        # makes alpha_prod_{-1} = 1
        alpha_prod_prev = jnp.where(
            timestep > 0,
            self.alphas_cumprod[timestep - 1],
            jnp.array(1.0, dtype=self.alphas_cumprod.dtype),
        )
        alpha_current = alpha_prod / alpha_prod_prev
        beta_prod = 1 - alpha_prod
        beta_prod_prev = 1 - alpha_prod_prev
        beta_current = 1 - alpha_current
        if prediction_type in ("sample", "x0"):
            original_sample = model_output
        elif prediction_type in ("error", "epsilon"):
            original_sample = (sample - beta_prod**0.5 * model_output) / (
                alpha_prod**0.5
            )
        else:
            raise NotImplementedError
        original_sample = jnp.clip(original_sample, -self.clamp_val, self.clamp_val)
        original_sample_coeff = (alpha_prod_prev**0.5 * beta_current) / beta_prod
        current_sample_coeff = alpha_current**0.5 * beta_prod_prev / beta_prod
        return sample * current_sample_coeff + original_sample * original_sample_coeff

    def q_variance(self, timestep):
        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = jnp.where(
            timestep > 0,
            self.alphas_cumprod[timestep - 1],
            jnp.array(1.0, dtype=self.alphas_cumprod.dtype),
        )
        alpha_current = alpha_prod / alpha_prod_prev
        beta_prod = 1 - alpha_prod
        beta_prod_prev = 1 - alpha_prod_prev
        beta_current = 1 - alpha_current
        variance = beta_prod_prev / beta_prod * beta_current
        return jnp.clip(variance, 1e-20)

    # def q_x0(self):
    #     pass TODO can i sample not directly but with some amount of steps?
