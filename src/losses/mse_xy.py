import equinox as eqx
import jax
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
        key,
        debug=False,
        **kwargs,
    ):
        future_local = AgentPath(
            future_path.path, future_path.action_len, ref_coords=past_path.ref_coords
        )
        gt_xy = (future_local.to_local()[..., :2]) / self.coord_scale

        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_xy.shape)
        noisy_xy = diffusion_sampler.add_noise(gt_xy, noise, timestep)
        pred_xy = model(timestep, noisy_xy, **kwargs)

        err = (pred_xy - gt_xy) ** 2
        agent_axes = tuple(range(1, err.ndim))
        weights = jnp.broadcast_to(
            jnp.any(future_path.path != 0, axis=-1, keepdims=True), gt_xy.shape
        )

        per_agent_loss = jnp.sum(err * weights, axis=agent_axes) / jnp.maximum(
            jnp.sum(weights, axis=agent_axes), 1.0
        )
        w = jnp.reshape(
            jnp.asarray(agents_coeffs, dtype=per_agent_loss.dtype), per_agent_loss.shape
        )
        loss = jnp.sum(per_agent_loss * w) / jnp.maximum(jnp.sum(w), 1.0)

        loss_dict = {"loss": loss}
        if debug:
            valid_weights = jnp.any(future_path.path != 0, axis=-1, keepdims=True)
            loss_dict.update(
                {
                    "noisy_abs_mean": masked_abs_mean(noisy_xy, valid_weights),
                    "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
                    "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
                    "valid_ratio": jnp.mean(valid_weights),
                }
            )
        return loss_dict
