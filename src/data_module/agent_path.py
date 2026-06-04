"""Agent trajectory path in local / global frames (Waymax feature order)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from src.utils.path_kinematics import inverse_kinematics, roll_out
from src.utils.trajectory_transform import wrap_angle


class AgentPath(eqx.Module):
    """Path tensor with trailing shape (A, T, 5): [x, y, yaw, vel_x, vel_y]."""

    path: jnp.ndarray
    action_len: int = eqx.field(static=True)
    ref_coords: jnp.ndarray = eqx.field(static=False)
    dt: float = eqx.field(static=True)
    num_timesteps: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        path: jnp.ndarray,
        action_len: int,
        ref_coords: jnp.ndarray | None = None,
        dt: float = 0.1,
    ):
        self.path = jnp.asarray(path)
        self.action_len = int(action_len)
        self.dt = float(dt)

        num_agents, num_timesteps, _ = self.path.shape
        self.num_timesteps = num_timesteps
        self.num_actions = 0 if num_timesteps <= 1 else num_timesteps // action_len

        if ref_coords is None:
            self.ref_coords = jnp.zeros((num_agents, 5))
        else:
            self.ref_coords = jnp.asarray(ref_coords)

    def to_local(self) -> jnp.ndarray:
        x, y, theta, v_x, v_y = [self.path[..., i] for i in range(5)]
        ref_x, ref_y, ref_theta = [self.ref_coords[:, i, None] for i in range(3)]

        dx, dy = x - ref_x, y - ref_y
        cos_t, sin_t = jnp.cos(ref_theta), jnp.sin(ref_theta)

        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        local_theta = wrap_angle(theta - ref_theta)
        local_v_x = v_x * cos_t + v_y * sin_t
        local_v_y = -v_x * sin_t + v_y * cos_t

        local_path = jnp.stack(
            [local_x, local_y, local_theta, local_v_x, local_v_y], axis=-1
        )
        valid = jnp.any(self.path != 0, axis=-1, keepdims=True)
        return jnp.where(valid, local_path, 0.0)

    def actions(self):
        valid = jnp.any(self.path[..., :2] != 0, axis=-1)
        return inverse_kinematics(self.to_local(), valid, self.action_len, self.dt)

    def decode_action_sample(
        self,
        sampled: jnp.ndarray,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ) -> jnp.ndarray:
        scale = jnp.asarray([accel_scale, yaw_rate_scale], dtype=sampled.dtype)
        rolled = roll_out(
            self.ref_coords,
            sampled * scale,
            self.action_len,
            self.dt,
            global_frame=False,
        )
        return rolled[..., :2]

    def decode_xy_sample(
        self,
        sampled: jnp.ndarray,
        coord_scale: float = 1.0,
        past_path: AgentPath | None = None,
    ) -> jnp.ndarray:
        base = (past_path if past_path is not None else self).to_local()[:, :1, :2]
        return (sampled * coord_scale) + base
