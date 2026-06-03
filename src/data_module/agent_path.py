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
    ref_idx: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    num_timesteps: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    valid_mask: jnp.ndarray | None = eqx.field(static=False)
    denoise_action_shape: tuple[int, ...] = eqx.field(static=True)
    denoise_xy_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        path: jnp.ndarray,
        action_len: int,
        ref_idx: int | None = None,
        valid_mask: jnp.ndarray | None = None,
        dt: float = 0.1,
    ):
        path = jnp.asarray(path)
        assert path.ndim == 2, "path should have shape (T,5) for a single agent"
        num_timesteps, _ = path.shape
        num_actions = 0 if num_timesteps <= 1 else num_timesteps // action_len

        self.path = path
        self.action_len = int(action_len)

        last_valid_idx = None
        if valid_mask is not None:
            vm = jnp.asarray(valid_mask)
            assert vm.shape == (num_timesteps,)
            idxs = jnp.arange(num_timesteps)
            last_valid_idx = int(jnp.max(jnp.where(vm, idxs, -1)))
        if ref_idx is None and last_valid_idx is not None:
            self.ref_idx = int(last_valid_idx)
        else:
            self.ref_idx = int(ref_idx) if ref_idx is not None else 0
        self.dt = float(dt)
        self.num_timesteps = int(num_timesteps)
        self.num_actions = int(num_actions)
        self.denoise_action_shape = (num_actions, 2)
        self.denoise_xy_shape = (num_timesteps, 2)
        self.valid_mask = jnp.asarray(valid_mask) if valid_mask is not None else None

    def denoise_shape(self, extract_actions: bool) -> tuple[int, ...]:
        return self.denoise_action_shape if extract_actions else self.denoise_xy_shape

    def to_local(self) -> jnp.ndarray:
        x = self.path[:, 0]
        y = self.path[:, 1]
        theta = self.path[:, 2]
        v_x = self.path[:, 3]
        v_y = self.path[:, 4]

        ri = int(self.ref_idx)
        ref_x = x[ri : ri + 1]
        ref_y = y[ri : ri + 1]
        ref_theta = theta[ri : ri + 1]

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
        valid = jnp.any(self.path[:, :5] != 0, axis=-1, keepdims=True)
        return jnp.where(valid, local_path, 0.0)

    def to_local_xy(self) -> jnp.ndarray:
        """Return only xy coordinates in local frame anchored at `ref_idx`."""
        return self.to_local()[:, :2]

    def local_xy(self) -> jnp.ndarray:
        xy = self.to_local()[:, :2]
        return xy - xy[:1, :]

    def actions(self, valid: jnp.ndarray):
        return inverse_kinematics(self.to_local(), valid, self.action_len, self.dt)

    def current_state_for_rollout(self) -> jnp.ndarray:
        local = self.to_local()
        ri = int(self.ref_idx)
        ref_state = local[ri, :]
        state = jnp.zeros_like(ref_state)
        return state.at[3:5].set(ref_state[3:5])

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
            return rel + self.to_local()[:1, :2]
        return rel + past_path.trajectory_from_anchor(self)[:1, :]

    def xy_to_global(self, local_xy: jnp.ndarray) -> jnp.ndarray:
        ri = int(self.ref_idx)
        anchor = self.path[ri]
        x0 = anchor[0]
        y0 = anchor[1]
        theta0 = anchor[2]
        cos_t = jnp.cos(theta0)
        sin_t = jnp.sin(theta0)
        g_x = local_xy[..., 0] * cos_t - local_xy[..., 1] * sin_t + x0
        g_y = local_xy[..., 0] * sin_t + local_xy[..., 1] * cos_t + y0
        if local_xy.shape[-1] == 2:
            return jnp.stack([g_x, g_y], axis=-1)
        g_theta = wrap_angle(local_xy[..., 2] + theta0)
        return jnp.stack([g_x, g_y, g_theta], axis=-1)

    def trajectory_from_anchor(self, other_path: AgentPath) -> jnp.ndarray:
        """Return `other_path` xy coordinates in the local frame anchored at `self.ref_idx`.

        The returned shape matches other_path.path[..., :2].
        """
        ri = int(self.ref_idx)
        anchor = self.path[ri, :]
        x = other_path.path[:, 0]
        y = other_path.path[:, 1]
        x0 = anchor[0]
        y0 = anchor[1]
        theta0 = anchor[2]
        cos_t = jnp.cos(theta0)
        sin_t = jnp.sin(theta0)
        dx = x - x0
        dy = y - y0
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        return jnp.stack([local_x, local_y], axis=-1)

    def actions_from_anchor(
        self,
        actions: jnp.ndarray,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ) -> jnp.ndarray:
        """Roll out `actions` starting from `self` anchor state and return xy trajectory relative to anchor."""
        scale = jnp.asarray([accel_scale, yaw_rate_scale], dtype=actions.dtype)
        start_state = self.current_state_for_rollout()
        rolled = roll_out(
            start_state, actions * scale, self.action_len, self.dt, global_frame=False
        )
        return rolled[..., :2]
