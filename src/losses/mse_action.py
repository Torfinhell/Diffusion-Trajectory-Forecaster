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
        agents_coeffs,
        key,
        debug=False,
        **kwargs,
    ):
        # ground-truth actions and validity
        gt_actions, actions_valid = future_path.actions()
        action_scale = jnp.asarray(
            [self.accel_scale, self.yaw_rate_scale], dtype=gt_actions.dtype
        )
        gt_actions_norm = gt_actions / action_scale

        # add noise and predict
        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_actions.shape)
        noisy_actions = diffusion_sampler.add_noise(gt_actions_norm, noise, timestep)
        pred_actions_norm = model(
            timestep, noisy_actions, **kwargs, past_path=past_path
        )

        # scale predictions back
        pred_actions = pred_actions_norm * action_scale

        # action-space MSE per-agent using actions_valid mask
        err_actions = (pred_actions - gt_actions) ** 2
        if err_actions.ndim < 1:
            raise ValueError("unexpected err_actions shape")
        agent_axes = tuple(range(1, err_actions.ndim))
        a_valid = actions_valid
        if a_valid.ndim == err_actions.ndim - 1:
            a_valid = a_valid[..., None]
        a_weights = jnp.asarray(a_valid, dtype=err_actions.dtype)
        a_weights = jnp.broadcast_to(a_weights, err_actions.shape)
        per_agent_num_a = jnp.sum(err_actions * a_weights, axis=agent_axes)
        per_agent_den_a = jnp.maximum(jnp.sum(a_weights, axis=agent_axes), 1.0)
        per_agent_mse_action = per_agent_num_a / per_agent_den_a

        # Aggregate across agents using provided agents_coeffs
        w = jnp.asarray(agents_coeffs, dtype=per_agent_mse_action.dtype)
        w = jnp.reshape(w, per_agent_mse_action.shape)
        loss = jnp.sum(per_agent_mse_action * w) / jnp.maximum(jnp.sum(w), 1.0)

        loss_dict = {"loss": loss}

        if debug:
            # XY diagnostics
            pred_xy = past_path.decode_action_sample(
                pred_actions_norm,
                accel_scale=self.accel_scale,
                yaw_rate_scale=self.yaw_rate_scale,
            )
            gt_xy = past_path.trajectory_from_anchor(future_path)
            xy_err = (pred_xy - gt_xy) ** 2
            valid_target = future_path.valid_mask
            if valid_target is None:
                valid_target = jnp.ones((future_path.path.shape[0],), dtype=bool)
            if valid_target.ndim == xy_err.ndim - 1:
                valid_target = valid_target[..., None]
            xy_weights = jnp.asarray(valid_target, dtype=xy_err.dtype)
            xy_weights = jnp.broadcast_to(xy_weights, xy_err.shape)

            loss_dict.update(
                {
                    "noisy_abs_mean": masked_abs_mean(
                        noisy_actions, a_weights[..., None]
                    ),
                    "target_abs_mean": masked_abs_mean(gt_xy, xy_weights),
                    "pred_abs_mean": masked_abs_mean(pred_xy, xy_weights),
                    "target_action_abs_mean": masked_abs_mean(
                        gt_actions, a_weights[..., None]
                    ),
                    "pred_action_abs_mean": masked_abs_mean(
                        pred_actions, a_weights[..., None]
                    ),
                    "valid_ratio": jnp.mean(xy_weights),
                }
            )

        return loss_dict
