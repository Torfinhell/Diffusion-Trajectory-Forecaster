import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from src.utils.data_utils import predictions_to_local_xy


def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)

class MSELoss(eqx.Module):
    accel_scale: float
    yaw_rate_scale: float

    def __init__(self, accel_scale: float = 1.0, yaw_rate_scale: float = 0.15):
        self.accel_scale = accel_scale
        self.yaw_rate_scale = yaw_rate_scale

    def __call__(
        self,
        model,
        diffusion_sampler,
        agent_past,
        agent_future,
        agents_coeffs,
        agent_future_valid,
        actions_future,
        actions_future_valid,
        key,
        debug=False,
        **kwargs,
    ):
        gt_actions = jnp.asarray(actions_future, dtype=jnp.float32)
        action_scale = jnp.asarray(
            [self.accel_scale, self.yaw_rate_scale], dtype=gt_actions.dtype
        )
        gt_actions_norm = gt_actions / action_scale

        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_actions.shape)
        noisy_actions = diffusion_sampler.add_noise(gt_actions_norm, noise, timestep)
        pred_actions_norm = model(timestep, noisy_actions, **kwargs)

        pred_xy, pred_actions = predictions_to_local_xy(
            pred_actions_norm,
            agent_past=agent_past,
            origin_vel=kwargs["origin_vel"],
            agent_future=agent_future,
            actions_future=actions_future,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        gt_xy = jnp.asarray(agent_future[..., :2], dtype=jnp.float32)

        err = (pred_xy - gt_xy) ** 2
        valid_target = agent_future_valid
        if valid_target.ndim == err.ndim - 1:
            valid_target = valid_target[..., None]
        weights = jnp.asarray(agents_coeffs, dtype=err.dtype)[..., None, None]
        weights = weights * jnp.asarray(valid_target, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        weighted_element_count = jnp.ones_like(err) * weights
        loss = (err * weights).sum() / jnp.maximum(weighted_element_count.sum(), 1.0)

        if debug:
            xy_valid_weights = jnp.asarray(valid_target, dtype=gt_xy.dtype)
            action_valid_weights = jnp.asarray(actions_future_valid, dtype=noisy_actions.dtype)
            if action_valid_weights.ndim == noisy_actions.ndim - 1:
                action_valid_weights = action_valid_weights[..., None]
            stats = {
                "noisy_abs_mean": masked_abs_mean(noisy_actions, action_valid_weights),
                "target_abs_mean": masked_abs_mean(gt_xy, xy_valid_weights),
                "pred_abs_mean": masked_abs_mean(pred_xy, xy_valid_weights),
                "target_action_abs_mean": masked_abs_mean(
                    gt_actions, action_valid_weights
                ),
                "pred_action_abs_mean": masked_abs_mean(
                    pred_actions, action_valid_weights
                ),
                "valid_ratio": jnp.mean(xy_valid_weights),
            }
            return loss, stats
        return loss
