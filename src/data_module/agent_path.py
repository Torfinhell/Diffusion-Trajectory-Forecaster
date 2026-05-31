"""Agent trajectory path in local / global frames (Waymax feature order)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from src.utils.path_kinematics import inverse_kinematics, roll_out
from src.utils.trajectory_transform import wrap_angle

PATH_FEATURE_DIM = 5


class AgentPath(eqx.Module):
    """Path tensor with trailing shape (A, T, 5): [x, y, yaw, vel_x, vel_y]."""

    path: jnp.ndarray
    action_len: int = eqx.field(static=True)
    ref_idx: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    num_agents: int = eqx.field(static=True)
    num_timesteps: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    denoise_action_shape: tuple[int, ...] = eqx.field(static=True)
    denoise_xy_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        path: jnp.ndarray,
        action_len: int,
        ref_idx: int = -1,
        dt: float = 0.1,
    ):
        path = jnp.asarray(path)
        if path.ndim != 3 or path.shape[-1] != PATH_FEATURE_DIM:
            raise ValueError(
                f"AgentPath expects (A, T, {PATH_FEATURE_DIM}), got {path.shape}"
            )
        num_agents, num_timesteps, _ = path.shape
        if num_timesteps > 1 and (num_timesteps - 1) % action_len != 0:
            raise ValueError("num_timesteps - 1 must be divisible by action_len")
        num_actions = 0 if num_timesteps <= 1 else (num_timesteps - 1) // action_len

        self.path = path
        self.action_len = int(action_len)
        self.ref_idx = int(ref_idx)
        self.dt = float(dt)
        self.num_agents = int(num_agents)
        self.num_timesteps = int(num_timesteps)
        self.num_actions = int(num_actions)
        self.denoise_action_shape = (num_agents, num_actions, 2)
        self.denoise_xy_shape = (num_agents, num_timesteps, 2)

    def denoise_shape(self, extract_actions: bool) -> tuple[int, ...]:
        return self.denoise_action_shape if extract_actions else self.denoise_xy_shape

    def to_local(self) -> jnp.ndarray:
        x = self.path[..., 0]
        y = self.path[..., 1]
        theta = self.path[..., 2]
        v_x = self.path[..., 3]
        v_y = self.path[..., 4]

        ref_x = x[..., self.ref_idx, None]
        ref_y = y[..., self.ref_idx, None]
        ref_theta = theta[..., self.ref_idx, None]
        cos_t = jnp.cos(ref_theta)
        sin_t = jnp.sin(ref_theta)

        dx = x - ref_x
        dy = y - ref_y
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        local_theta = wrap_angle(theta - ref_theta)
        local_v_x = v_x * cos_t + v_y * sin_t
        local_v_y = -v_x * sin_t + v_y * cos_t

        local_path = jnp.stack(
            [local_x, local_y, local_theta, local_v_x, local_v_y], axis=-1
        )
        valid = jnp.any(self.path[..., :5] != 0, axis=-1, keepdims=True)
        return jnp.where(valid, local_path, 0.0)

    def local_xy(self) -> jnp.ndarray:
        xy = self.to_local()[..., :2]
        return xy - xy[..., :1, :]

    def actions(self, valid: jnp.ndarray):
        return inverse_kinematics(self.to_local(), valid, self.action_len, self.dt)

    def current_state_for_rollout(self) -> jnp.ndarray:
        ref_state = self.to_local()[..., self.ref_idx, :]
        state = jnp.zeros_like(ref_state)
        return state.at[..., 3:5].set(ref_state[..., 3:5])

    def rollout_actions(
        self,
        actions: jnp.ndarray,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ) -> jnp.ndarray:
        scale = jnp.asarray([accel_scale, yaw_rate_scale], dtype=actions.dtype)
        rolled = roll_out(
            self.current_state_for_rollout(),
            actions * scale,
            self.action_len,
            self.dt,
            global_frame=False,
        )
        return rolled[..., :2]

    def decode_action_sample(
        self,
        sampled: jnp.ndarray,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ) -> jnp.ndarray:
        return self.rollout_actions(sampled, accel_scale, yaw_rate_scale)

    def decode_xy_sample(
        self,
        sampled: jnp.ndarray,
        coord_scale: float = 1.0,
        past_path: AgentPath | None = None,
    ) -> jnp.ndarray:
        rel = sampled * coord_scale
        if past_path is None:
            return rel + self.to_local()[..., :1, :2]
        return rel + past_path.future_xy_in_past_frame(self)[..., :1, :]

    def xy_to_global(self, local_xy: jnp.ndarray) -> jnp.ndarray:
        anchor = self.path[..., self.ref_idx, :]
        x0 = anchor[..., 0]
        y0 = anchor[..., 1]
        theta0 = anchor[..., 2]
        cos_t = jnp.cos(theta0)[..., None]
        sin_t = jnp.sin(theta0)[..., None]
        g_x = local_xy[..., 0] * cos_t - local_xy[..., 1] * sin_t + x0[..., None]
        g_y = local_xy[..., 0] * sin_t + local_xy[..., 1] * cos_t + y0[..., None]
        if local_xy.shape[-1] == 2:
            return jnp.stack([g_x, g_y], axis=-1)
        g_theta = wrap_angle(local_xy[..., 2] + theta0[..., None])
        return jnp.stack([g_x, g_y, g_theta], axis=-1)

    def future_xy_in_past_frame(self, past_path: AgentPath) -> jnp.ndarray:
        anchor = past_path.path[..., -1, :]
        x = self.path[..., 0]
        y = self.path[..., 1]
        x0 = anchor[..., 0][..., None]
        y0 = anchor[..., 1][..., None]
        theta0 = anchor[..., 2][..., None]
        cos_t = jnp.cos(theta0)
        sin_t = jnp.sin(theta0)
        dx = x - x0
        dy = y - y0
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        return jnp.stack([local_x, local_y], axis=-1)

    def actions_for_encoder(self, valid: jnp.ndarray) -> jnp.ndarray:
        actions, _ = self.actions(valid)
        return actions
