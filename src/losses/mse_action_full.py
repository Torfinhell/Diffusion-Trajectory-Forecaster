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
        agents_coeffs,
        key,
        debug: bool = False,
        **kwargs,
    ):
        gt_actions, actions_valid = future_path.actions()
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

        err_actions = (pred_actions - gt_actions) ** 2
        agent_axes = tuple(range(1, err_actions.ndim))
        a_weights = jnp.broadcast_to(
            jnp.asarray(actions_valid, dtype=err_actions.dtype)[..., None],
            err_actions.shape,
        )
        per_agent_mse_action = jnp.sum(
            err_actions * a_weights, axis=agent_axes
        ) / jnp.maximum(jnp.sum(a_weights, axis=agent_axes), 1.0)

        pred_full_xy = past_path.decode_action_sample(
            pred_actions_norm,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )

        dx = pred_full_xy[..., 0] - past_path.ref_coords[:, 0, None]
        dy = pred_full_xy[..., 1] - past_path.ref_coords[:, 1, None]
        cos_t = jnp.cos(past_path.ref_coords[:, 2, None])
        sin_t = jnp.sin(past_path.ref_coords[:, 2, None])
        pred_xy_local = jnp.stack(
            [dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t], axis=-1
        )
        pred_xy_local = jnp.where(
            jnp.any(future_path.path != 0, axis=-1, keepdims=True), pred_xy_local, 0.0
        )

        future_local = AgentPath(
            future_path.path, future_path.action_len, ref_coords=past_path.ref_coords
        )
        gt_xy = future_local.to_local()[..., :2]

        err_xy = (pred_xy_local - gt_xy) ** 2
        agent_axes_xy = tuple(range(1, err_xy.ndim))
        v_weights = jnp.broadcast_to(
            jnp.any(future_path.path != 0, axis=-1, keepdims=True), gt_xy.shape
        )
        per_agent_num_xy = jnp.sum(err_xy * v_weights, axis=agent_axes_xy)
        per_agent_den_xy = jnp.maximum(jnp.sum(v_weights, axis=agent_axes_xy), 1.0)
        per_agent_mse_xy_full = per_agent_num_xy / per_agent_den_xy

        w = jnp.reshape(
            jnp.asarray(agents_coeffs, dtype=per_agent_mse_action.dtype),
            per_agent_mse_action.shape,
        )
        mse_action = jnp.sum(per_agent_mse_action * w) / jnp.maximum(jnp.sum(w), 1.0)
        mse_xy_full = jnp.sum(per_agent_mse_xy_full * w) / jnp.maximum(jnp.sum(w), 1.0)

        return {"loss": mse_action, "mse_xy_full": mse_xy_full}
