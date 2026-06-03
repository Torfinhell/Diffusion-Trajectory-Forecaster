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
    ref_idx: jnp.ndarray | int = eqx.field(static=False)
    dt: float = eqx.field(static=True)
    num_timesteps: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    valid_mask: jnp.ndarray | None = eqx.field(static=False)

    def __init__(
        self,
        path: jnp.ndarray,
        action_len: int,
        ref_idx: int | None = None,
        valid_mask: jnp.ndarray | None = None,
        dt: float = 0.1,
    ):
        path = jnp.asarray(path)
        assert path.ndim == 3, "path should have shape (A,T,5)"
        num_agents, num_timesteps, _ = path.shape
        num_actions = 0 if num_timesteps <= 1 else num_timesteps // action_len

        self.path = path
        self.action_len = int(action_len)

        if valid_mask is not None:
            vm = jnp.asarray(valid_mask)
            assert vm.shape == (num_agents, num_timesteps)
        else:
            vm = None

        if ref_idx is None:
            if vm is not None:
                idxs = jnp.arange(num_timesteps)
                last_valid_idx = jnp.max(jnp.where(vm, idxs[None, :], -1), axis=1)
            else:
                last_valid_idx = jnp.zeros((num_agents,), dtype=int)
            self.ref_idx = last_valid_idx
        else:
            if isinstance(ref_idx, (int,)):
                self.ref_idx = jnp.full((num_agents,), int(ref_idx), dtype=int)
            else:
                self.ref_idx = jnp.asarray(ref_idx, dtype=int)

        self.dt = float(dt)
        self.num_timesteps = int(num_timesteps)
        self.num_actions = int(num_actions)
        self.valid_mask = jnp.asarray(valid_mask) if valid_mask is not None else None

    def denoise_shape(self, extract_actions: bool) -> tuple[int, ...]:
        if extract_actions:
            return (self.path.shape[0], self.num_actions, 2)
        return (self.path.shape[0], self.num_timesteps, 2)

    def to_local(self) -> jnp.ndarray:
        x = self.path[..., 0]
        y = self.path[..., 1]
        theta = self.path[..., 2]
        v_x = self.path[..., 3]
        v_y = self.path[..., 4]

        ri = self.ref_idx
        ref_x = jnp.take_along_axis(x, ri[:, None], axis=1)
        ref_y = jnp.take_along_axis(y, ri[:, None], axis=1)
        ref_theta = jnp.take_along_axis(theta, ri[:, None], axis=1)

        dx = x - ref_x
        dy = y - ref_y
        cos_t = jnp.cos(ref_theta)
        sin_t = jnp.sin(ref_theta)
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

    def to_local_xy(self) -> jnp.ndarray:
        """Return only xy coordinates in local frame anchored at `ref_idx`."""
        return self.to_local()[..., :2]

    def local_xy(self) -> jnp.ndarray:
        xy = self.to_local()[..., :2]
        return xy - xy[:, :1, :]

    def actions(self):
        valid = self.valid_mask
        if valid is None:
            valid = jnp.any(self.path[..., :2] != 0, axis=-1)
        return inverse_kinematics(self.to_local(), valid, self.action_len, self.dt)

    def current_state_for_rollout(self) -> jnp.ndarray:
        local = self.to_local()
        ri = self.ref_idx
        ref_state = jnp.take_along_axis(local, ri[:, None, None], axis=1)[:, 0, :]
        state = jnp.zeros_like(ref_state)
        return state.at[:, 3:5].set(ref_state[:, 3:5])

    def decode_action_sample(
        self,
        sampled: jnp.ndarray,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ) -> jnp.ndarray:
        scale = jnp.asarray([accel_scale, yaw_rate_scale], dtype=sampled.dtype)
        rolled = roll_out(
            self.current_state_for_rollout(),
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
        rel = sampled * coord_scale
        if past_path is None:
            return rel + self.to_local()[:, :1, :2]
        return rel + past_path.trajectory_from_anchor(self)

    def xy_to_global(self, local_xy: jnp.ndarray) -> jnp.ndarray:
        ri = self.ref_idx
        anchor = jnp.take_along_axis(self.path, ri[:, None, None], axis=1)[:, 0, :]
        x0 = anchor[..., 0]
        y0 = anchor[..., 1]
        theta0 = anchor[..., 2]
        cos_t = jnp.cos(theta0)
        sin_t = jnp.sin(theta0)
        g_x = (
            local_xy[..., 0] * cos_t[..., None]
            - local_xy[..., 1] * sin_t[..., None]
            + x0[..., None]
        )
        g_y = (
            local_xy[..., 0] * sin_t[..., None]
            + local_xy[..., 1] * cos_t[..., None]
            + y0[..., None]
        )
        if local_xy.shape[-1] == 2:
            return jnp.stack([g_x, g_y], axis=-1)
        g_theta = wrap_angle(local_xy[..., 2] + theta0[..., None])
        return jnp.stack([g_x, g_y, g_theta], axis=-1)

    def trajectory_from_anchor(self, other_path: AgentPath) -> jnp.ndarray:
        """Return `other_path` xy coordinates in the local frame anchored at `self.ref_idx`.

        The returned shape matches other_path.path[..., :2].
        """
        ri = self.ref_idx
        anchor = jnp.take_along_axis(self.path, ri[:, None, None], axis=1)[:, 0, :]
        x = other_path.path[..., 0]
        y = other_path.path[..., 1]
        x0 = anchor[..., 0]
        y0 = anchor[..., 1]
        theta0 = anchor[..., 2]
        cos_t = jnp.cos(theta0)
        sin_t = jnp.sin(theta0)
        dx = x - x0[..., None]
        dy = y - y0[..., None]
        local_x = dx * cos_t[..., None] + dy * sin_t[..., None]
        local_y = -dx * sin_t[..., None] + dy * cos_t[..., None]
        return jnp.stack([local_x, local_y], axis=-1)
