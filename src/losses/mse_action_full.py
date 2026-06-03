import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.data_module.agent_path import AgentPath


class MseActionFullLoss(eqx.Module):
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
        debug: bool = False,
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
        pred_actions = pred_actions_norm * action_scale

        # action-space MSE per-agent
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

        # decode to full trajectory (xy) from predicted actions using past_path as anchor
        pred_full_xy = past_path.trajectory_from_actions(
            pred_actions,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        gt_xy = past_path.trajectory_from_anchor(future_path)
        err_xy = (pred_full_xy - gt_xy) ** 2
        if err_xy.ndim < 1:
            raise ValueError("unexpected err_xy shape")
        agent_axes_xy = tuple(range(1, err_xy.ndim))
        v = valid
        if v.ndim == err_xy.ndim - 1:
            v = v[..., None]
        v_weights = jnp.asarray(v, dtype=err_xy.dtype)
        v_weights = jnp.broadcast_to(v_weights, err_xy.shape)
        per_agent_num_xy = jnp.sum(err_xy * v_weights, axis=agent_axes_xy)
        per_agent_den_xy = jnp.maximum(jnp.sum(v_weights, axis=agent_axes_xy), 1.0)
        per_agent_mse_xy_full = per_agent_num_xy / per_agent_den_xy

        # `agent_coeffs` is required and should be provided by the caller
        w = jnp.asarray(agent_coeffs, dtype=per_agent_mse_action.dtype)
        w = jnp.reshape(w, per_agent_mse_action.shape)
        mse_action = jnp.sum(per_agent_mse_action * w) / jnp.maximum(jnp.sum(w), 1.0)
        w_xy = jnp.asarray(agent_coeffs, dtype=per_agent_mse_xy_full.dtype)
        w_xy = jnp.reshape(w_xy, per_agent_mse_xy_full.shape)
        mse_xy_full = jnp.sum(per_agent_mse_xy_full * w_xy) / jnp.maximum(
            jnp.sum(w_xy), 1.0
        )

        loss_dict = {"loss": mse_action}
        loss_dict.update({"mse_xy_full": mse_xy_full})
        return loss_dict
