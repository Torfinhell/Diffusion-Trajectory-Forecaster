import equinox as eqx
import jax
import jax.numpy as jnp


class OldSampler(eqx.Module):
    num_steps: int
    clamp_val: float
    t0: float
    t1: float
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    times: jax.Array
    schedule: str

    def __init__(
        self,
        steps=100,
        schedule="vp_linear",
        clamp_val=100.0,
        t0=1e-3,
        t1=2.0,
    ):
        self.schedule = schedule
        self.clamp_val = clamp_val
        self.num_steps = steps
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.times = jnp.linspace(self.t0, self.t1, self.num_steps)
        alpha_t, _ = self.alpha_sigma(self.times)
        self.alphas_cumprod = alpha_t**2
        alpha_prev = jnp.concatenate(
            [jnp.ones((1,), dtype=alpha_t.dtype), self.alphas_cumprod[:-1]]
        )
        self.alphas = self.alphas_cumprod / alpha_prev
        self.betas = 1.0 - self.alphas

    def int_beta(self, t):
        if self.schedule == "vp_linear":
            return t
        raise NotImplementedError("Old schedule is not defined")

    def alpha_sigma(self, t):
        int_beta_t = self.int_beta(t)
        alpha = jnp.exp(-0.5 * int_beta_t)
        sigma = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-int_beta_t), 1e-5))
        return alpha, sigma

    def add_noise(
        self,
        x_0,  # (F, T)
        noise,  # (F, T)
        timestep,  # (1,)
    ):
        alpha_t, sigma_t = self.alpha_sigma(self.times[timestep])
        return alpha_t * x_0 + sigma_t * noise

    def step(
        self,
        key,
        model_output,  # (T, F)
        timestep,  # (1,)
        sample,  # (T, F)
        prediction_type="sample",
    ):
        del key
        if prediction_type not in ["sample", "error", "x0", "epsilon", "score"]:
            raise ValueError(f"Invalid prediction_type: {prediction_type}")

        alpha_t, sigma_t = self.alpha_sigma(self.times[timestep])
        prev_time = jnp.where(timestep > 0, self.times[timestep - 1], self.times[0])
        alpha_prev, sigma_prev = self.alpha_sigma(prev_time)

        if prediction_type in ("sample", "x0"):
            original_sample = model_output
            pred_epsilon = (sample - alpha_t * original_sample) / jnp.maximum(
                sigma_t, 1e-5
            )
        elif prediction_type in ("error", "epsilon"):
            pred_epsilon = model_output
            original_sample = (sample - sigma_t * pred_epsilon) / jnp.maximum(
                alpha_t, 1e-5
            )
        elif prediction_type == "score":
            pred_epsilon = -sigma_t * model_output
            original_sample = (sample + (sigma_t**2) * model_output) / jnp.maximum(
                alpha_t, 1e-5
            )
        else:
            raise NotImplementedError

        original_sample = jnp.clip(original_sample, -self.clamp_val, self.clamp_val)
        return alpha_prev * original_sample + sigma_prev * pred_epsilon
