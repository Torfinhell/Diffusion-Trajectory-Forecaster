import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from src.data_module.agent_path import AgentPath


def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)


class MseXYLoss(eqx.Module):
    coord_scale: float

    def __init__(self, coord_scale: float = 1.0):
        self.coord_scale = coord_scale

    def __call__(
        self,
        model,
        diffusion_sampler,
        past_path: AgentPath,
        future_path: AgentPath,
        agents_coeffs,
        agent_future_valid,
        key,
        debug=False,
        **kwargs,
    ):
        gt_xy = future_path.local_xy() / self.coord_scale
        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_xy.shape)
        y = diffusion_sampler.add_noise(gt_xy, noise, timestep)
        pred_xy = model(timestep, y, **kwargs)
        err = (pred_xy - gt_xy) ** 2
        weights = jnp.asarray(agents_coeffs, dtype=err.dtype)[
            ..., None, None
        ] * jnp.asarray(agent_future_valid, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        loss = (err * weights).sum() / jnp.maximum(
            (jnp.ones_like(err) * weights).sum(), 1.0
        )
        loss_dict = {"loss": loss}
        if debug:
            valid_weights = jnp.asarray(agent_future_valid, dtype=gt_xy.dtype)
            loss_dict.update(
                {
                    "noisy_abs_mean": masked_abs_mean(y, valid_weights),
                    "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
                    "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
                    "valid_ratio": jnp.mean(valid_weights),
                }
            )
        return loss_dict
