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
        agent_coeffs,
        key,
        debug=False,
        **kwargs,
    ):
        valid = future_path.valid_mask
        if valid is None:
            valid = jnp.ones((future_path.path.shape[0],), dtype=bool)
        gt_xy = past_path.trajectory_from_anchor(future_path) / self.coord_scale
        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_xy.shape)
        y = diffusion_sampler.add_noise(gt_xy, noise, timestep)
        pred_xy = model(timestep, y, **kwargs)
        err = (pred_xy - gt_xy) ** 2
        if err.ndim < 1:
            raise ValueError("unexpected err shape")
        agent_axes = tuple(range(1, err.ndim))
        valid_target = valid
        if valid_target.ndim == err.ndim - 1:
            valid_target = valid_target[..., None]
        weights = jnp.asarray(valid_target, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        per_agent_num = jnp.sum(err * weights, axis=agent_axes)
        per_agent_den = jnp.maximum(jnp.sum(weights, axis=agent_axes), 1.0)
        per_agent_loss = per_agent_num / per_agent_den

        # `agent_coeffs` is required and should be provided by the caller
        w = jnp.asarray(agent_coeffs, dtype=per_agent_loss.dtype)
        w = jnp.reshape(w, per_agent_loss.shape)
        loss = jnp.sum(per_agent_loss * w) / jnp.maximum(jnp.sum(w), 1.0)
        loss_dict = {"loss": loss}
        if debug:
            valid_weights = jnp.asarray(valid, dtype=gt_xy.dtype)
            loss_dict.update(
                {
                    "noisy_abs_mean": masked_abs_mean(y, valid_weights),
                    "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
                    "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
                    "valid_ratio": jnp.mean(valid_weights),
                }
            )
        return loss_dict
