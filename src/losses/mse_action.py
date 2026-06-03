import equinox as eqx
import jax
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
        agent_coeffs,
        key,
        debug=False,
        **kwargs,
    ):
        valid = future_path.valid_mask
        if valid is None:
            valid = jnp.ones((future_path.path.shape[0],), dtype=bool)
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
        gt_xy = past_path.trajectory_from_anchor(future_path)

        err = (pred_xy - gt_xy) ** 2
        # per-agent normalization: sum over non-agent axes, producing shape (num_agents,)
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
