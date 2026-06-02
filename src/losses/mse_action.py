import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from src.data_module.agent_path import AgentPath


def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)


class MseActionLoss(eqx.Module):
    accel_scale: float
    yaw_rate_scale: float

    def __init__(self, accel_scale: float = 1.0, yaw_rate_scale: float = 0.15):
        self.accel_scale = accel_scale
        self.yaw_rate_scale = yaw_rate_scale

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
        valid = agent_future_valid
        if valid.ndim == future_path.path.ndim:
            valid = valid[..., 0]
        gt_actions, actions_valid = future_path.actions(valid)
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
        pred_actions_norm = model(
            timestep, noisy_actions, **kwargs, past_path=past_path
        )
        pred_xy = past_path.decode_action_sample(
            pred_actions_norm,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        pred_actions = pred_actions_norm * action_scale
        gt_xy = future_path.future_xy_in_past_frame(past_path)

        err = (pred_xy - gt_xy) ** 2
        valid_target = agent_future_valid
        if valid_target.ndim == err.ndim - 1:
            valid_target = valid_target[..., None]
        weights = jnp.asarray(agents_coeffs, dtype=err.dtype)[..., None, None]
        weights = weights * jnp.asarray(valid_target, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        loss = (err * weights).sum() / jnp.maximum(
            (jnp.ones_like(err) * weights).sum(), 1.0
        )
        loss_dict = {"loss": loss}
        if debug:
            xy_valid_weights = jnp.asarray(valid_target, dtype=gt_xy.dtype)
            action_valid_weights = jnp.asarray(actions_valid, dtype=noisy_actions.dtype)
            loss_dict.update(
                {
                    "noisy_abs_mean": masked_abs_mean(
                        noisy_actions, action_valid_weights[..., None]
                    ),
                    "target_abs_mean": masked_abs_mean(gt_xy, xy_valid_weights),
                    "pred_abs_mean": masked_abs_mean(pred_xy, xy_valid_weights),
                    "target_action_abs_mean": masked_abs_mean(
                        gt_actions, action_valid_weights[..., None]
                    ),
                    "pred_action_abs_mean": masked_abs_mean(
                        pred_actions, action_valid_weights[..., None]
                    ),
                    "valid_ratio": jnp.mean(xy_valid_weights),
                }
            )
        return loss_dict
