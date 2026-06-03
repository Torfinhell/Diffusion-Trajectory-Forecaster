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
        key,
        debug: bool = False,
        agent_coeff=None,
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

        # action-space MSE
        err_actions = (pred_actions - gt_actions) ** 2
        a_valid = actions_valid
        if a_valid.ndim == err_actions.ndim - 1:
            a_valid = a_valid[..., None]
        a_weights = jnp.asarray(a_valid, dtype=err_actions.dtype)
        a_weights = jnp.broadcast_to(a_weights, err_actions.shape)
        mse_action = (err_actions * a_weights).sum() / jnp.maximum(a_weights.sum(), 1.0)

        # decode to full trajectory (xy) from predicted actions using past_path as anchor
        pred_full_xy = past_path.trajectory_from_actions(
            pred_actions,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        gt_xy = past_path.trajectory_from_anchor(future_path)
        err_xy = (pred_full_xy - gt_xy) ** 2
        v = valid
        if v.ndim == err_xy.ndim - 1:
            v = v[..., None]
        v_weights = jnp.asarray(v, dtype=err_xy.dtype)
        v_weights = jnp.broadcast_to(v_weights, err_xy.shape)
        mse_xy_full = (err_xy * v_weights).sum() / jnp.maximum(v_weights.sum(), 1.0)

        loss_dict = {"loss": mse_action}
        loss_dict.update({"mse_xy_full": mse_xy_full})
        # scale by agent coefficient if provided
        agent_coeff = (
            kwargs.pop("agent_coeff", agent_coeff)
            if "agent_coeff" in kwargs
            else agent_coeff
        )
        if agent_coeff is not None:
            loss_dict = jax.tree_map(
                lambda v: v * jnp.asarray(agent_coeff, dtype=v.dtype), loss_dict
            )
        return loss_dict
