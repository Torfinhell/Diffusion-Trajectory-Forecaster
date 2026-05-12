import jax
import jax.numpy as jnp


def denormalize_traj(traj, mean, var):
    if mean.ndim == 1:
        mean = mean[None, None, :]
        std = jnp.sqrt(jnp.maximum(var, 1e-6))[None, None, :]
    else:
        mean = mean[:, None, None, :]
        std = jnp.sqrt(jnp.maximum(var, 1e-6))[:, None, None, :]
    return traj * std + mean


def wrap_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jax.jit(
    static_argnames=["x_index", "y_index", "heading_index", "vel_x_index", "vel_y_index"]
)
def batch_transform_trajs_to_local_frame(
    trajs,
    origin_xy,
    origin_theta,
    x_index=0,
    y_index=1,
    heading_index=5,
    vel_x_index=3,
    vel_y_index=4,
):
    if trajs.ndim == 3:
        trajs = trajs[None, ...]
        squeeze_batch = True
    elif trajs.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_trajs_to_local_frame expected 3D or 4D input, "
            f"got {trajs.shape}"
        )

    x = trajs[..., x_index]  # (B, num_agents, num_timesteps)
    y = trajs[..., y_index]
    origin_xy = origin_xy[None, ...]      # (1, num_agents)
    origin_theta = origin_theta[None, ...] 
    origin_x = origin_xy[..., 0][..., None]# (1, num_agents, 1)
    origin_y = origin_xy[..., 1][..., None]
    origin_theta = origin_theta[..., None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)
    dx = x - origin_x
    dy = y - origin_y
    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta

    transformed = trajs.at[..., x_index].set(local_x)
    transformed = transformed.at[..., y_index].set(local_y)

    if heading_index is not None and trajs.shape[-1] > heading_index:
        theta = trajs[..., heading_index]
        local_theta = wrap_angle(theta - origin_theta)
        transformed = transformed.at[..., heading_index].set(local_theta)

    if (
        vel_x_index is not None
        and vel_y_index is not None
        and trajs.shape[-1] > max(vel_x_index, vel_y_index)
    ):
        v_x = trajs[..., vel_x_index]
        v_y = trajs[..., vel_y_index]
        local_v_x = v_x * cos_theta + v_y * sin_theta
        local_v_y = -v_x * sin_theta + v_y * cos_theta
        transformed = transformed.at[..., vel_x_index].set(local_v_x)
        transformed = transformed.at[..., vel_y_index].set(local_v_y)

    valid_mask = jnp.any(trajs[..., :3] != 0, axis=-1, keepdims=True)
    transformed = jnp.where(valid_mask, transformed, 0.0)
    if squeeze_batch:
        return transformed[0]
    return transformed


def batch_transform_polylines_to_local_frame(polylines):
    if polylines.ndim == 3:
        polylines = polylines[None, ...]
        squeeze_batch = True
    elif polylines.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_polylines_to_local_frame expected 3D or 4D input, "
            f"got {polylines.shape}"
        )

    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]
    origin_x = x[:, :, 0, None]
    origin_y = y[:, :, 0, None]
    origin_theta = theta[:, :, 0, None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)
    dx = x - origin_x
    dy = y - origin_y
    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta
    local_theta = wrap_angle(theta - origin_theta)
    local_polylines = jnp.stack([local_x, local_y, local_theta], axis=-1)
    valid_mask = jnp.any(polylines[..., :3] != 0, axis=-1, keepdims=True)
    local_polylines = jnp.where(valid_mask, local_polylines, 0.0)
    transformed = jnp.concatenate([local_polylines, polylines[..., 3:]], axis=-1)

    origin_info = jnp.concatenate([origin_x, origin_y, origin_theta], axis=-1)
    if squeeze_batch:
        return transformed[0], origin_info[0]
    return transformed, origin_info

@jax.jit(
    static_argnames=["x_index", "y_index", "heading_index", "vel_x_index", "vel_y_index"]
)
def batch_transform_trajs_to_global_frame(
    trajs,
    origin_xy,
    origin_theta,
    x_index=0,
    y_index=1,
    heading_index=None,
    vel_x_index=3,
    vel_y_index=4,
):
    if trajs.ndim == 3:
        trajs = trajs[None, ...]
        origin_xy = origin_xy[None, ...]
        origin_theta = origin_theta[None, ...]
        squeeze_batch = True
    elif trajs.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_trajs_to_global_frame expected 3D or 4D input, "
            f"got {trajs.shape}"
        )

    local_x = trajs[..., x_index]
    local_y = trajs[..., y_index]
    origin_x = origin_xy[..., 0][..., None]
    origin_y = origin_xy[..., 1][..., None]
    origin_theta = origin_theta[..., None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)

    global_x = local_x * cos_theta - local_y * sin_theta + origin_x
    global_y = local_x * sin_theta + local_y * cos_theta + origin_y

    transformed = trajs.at[..., x_index].set(global_x)
    transformed = transformed.at[..., y_index].set(global_y)

    if heading_index is not None and trajs.shape[-1] > heading_index:
        global_theta = wrap_angle(trajs[..., heading_index] + origin_theta)
        transformed = transformed.at[..., heading_index].set(global_theta)

    if (
        vel_x_index is not None
        and vel_y_index is not None
        and trajs.shape[-1] > max(vel_x_index, vel_y_index)
    ):
        local_v_x = trajs[..., vel_x_index]
        local_v_y = trajs[..., vel_y_index]
        global_v_x = local_v_x * cos_theta - local_v_y * sin_theta
        global_v_y = local_v_x * sin_theta + local_v_y * cos_theta
        transformed = transformed.at[..., vel_x_index].set(global_v_x)
        transformed = transformed.at[..., vel_y_index].set(global_v_y)

    if squeeze_batch:
        return transformed[0]
    return transformed


@jax.jit
def batch_transform_polylines_to_global_frame(
    polylines,
    origin_xy,
    origin_theta,
):
    if polylines.ndim == 3:
        polylines = polylines[None, ...]
        origin_xy = origin_xy[None, ...]
        origin_theta = origin_theta[None, ...]
        squeeze_batch = True
    elif polylines.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_polylines_to_global_frame expected 3D or 4D input, "
            f"got {polylines.shape}"
        )

    local_x = polylines[..., 0]
    local_y = polylines[..., 1]
    local_theta = polylines[..., 2]
    origin_x = origin_xy[..., 0][..., None]
    origin_y = origin_xy[..., 1][..., None]
    origin_theta = origin_theta[..., None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)

    global_x = local_x * cos_theta - local_y * sin_theta + origin_x
    global_y = local_x * sin_theta + local_y * cos_theta + origin_y
    global_theta = wrap_angle(local_theta + origin_theta)

    global_polylines = jnp.stack([global_x, global_y, global_theta], axis=-1)
    valid_mask = jnp.any(polylines[..., :3] != 0, axis=-1, keepdims=True)
    global_polylines = jnp.where(valid_mask, global_polylines, 0.0)
    transformed = jnp.concatenate([global_polylines, polylines[..., 3:]], axis=-1)
    if squeeze_batch:
        return transformed[0]
    return transformed


def inverse_kinematics(
    agents_future: jnp.ndarray,
    agents_future_valid: jnp.ndarray,
    dt: float = 0.1,
    action_len: int = 5,
    x_index: int = 0,
    y_index: int = 1,
    vel_x_index: int = 3,
    vel_y_index: int = 4,
    yaw_index: int = 5,
):
    """
    Perform inverse kinematics to compute actions.

    Args:
        agents_future: Future agent states tensor with trailing shape [..., T, F].
            Defaults assume repo feature order [x, y, z, vel_x, vel_y, yaw].
        agents_future_valid: Future validity mask with trailing shape [..., T]
            or [..., T, 1].
        dt (float): Time interval. Default is 0.1.
        action_len (int): Length of each action. Default is 5.

    Returns:
        Tuple of:
        - actions with shape [..., num_actions, 2] storing [accel, yaw_rate]
        - action_valid with shape [..., num_actions]
    """
    agents_future = jnp.asarray(agents_future)
    agents_future_valid = jnp.asarray(agents_future_valid)
    if agents_future_valid.ndim == agents_future.ndim:
        agents_future_valid = agents_future_valid[..., 0]

    num_timesteps = agents_future.shape[-2]
    if (num_timesteps - 1) % action_len != 0:
        raise ValueError("future_len - 1 must be divisible by action_len")
    num_actions = (num_timesteps - 1) // action_len

    yaw = agents_future[..., yaw_index]
    vel_x = agents_future[..., vel_x_index]
    vel_y = agents_future[..., vel_y_index]
    speed = jnp.sqrt(vel_x**2 + vel_y**2)

    yaw_rate = wrap_angle(jnp.diff(yaw, axis=-1)) / dt
    accel = jnp.diff(speed, axis=-1) / dt
    action_valid = agents_future_valid[..., :-1] & agents_future_valid[..., 1:]

    yaw_rate = jnp.where(action_valid, yaw_rate, 0.0)
    accel = jnp.where(action_valid, accel, 0.0)

    prefix_shape = agents_future.shape[:-2]
    yaw_rate = yaw_rate.reshape(*prefix_shape, num_actions, action_len)
    accel = accel.reshape(*prefix_shape, num_actions, action_len)
    action_valid = action_valid.reshape(*prefix_shape, num_actions, action_len)

    valid_count = jnp.maximum(action_valid.sum(axis=-1), 1)
    yaw_rate_sample = yaw_rate.sum(axis=-1) / valid_count
    accel_sample = accel.sum(axis=-1) / valid_count
    action = jnp.stack([accel_sample, yaw_rate_sample], axis=-1)
    action_valid = jnp.any(action_valid, axis=-1)
    action = jnp.where(action_valid[..., None], action, 0.0)
    return action, action_valid


def make_local_rollout_state(agent_past: jnp.ndarray, origin_vel: jnp.ndarray):
    current_state = jnp.zeros_like(agent_past[..., 0, :])
    return current_state.at[..., 3:5].set(origin_vel)


def predictions_to_local_xy(
    sampled_pred: jnp.ndarray,
    agent_past: jnp.ndarray,
    origin_vel: jnp.ndarray,
    agent_future: jnp.ndarray,
    actions_future: jnp.ndarray,
    accel_scale: float,
    yaw_rate_scale: float,
):
    scale = jnp.asarray([accel_scale, yaw_rate_scale], dtype=sampled_pred.dtype)
    pred_actions = sampled_pred * scale
    current_state = make_local_rollout_state(agent_past, origin_vel)
    action_len = agent_future.shape[-2] // actions_future.shape[-2]
    pred_rollout = roll_out(
        current_state,
        pred_actions,
        action_len=action_len,
        global_frame=False,
    )
    return pred_rollout[..., :2], pred_actions


def roll_out(
    current_states: jnp.ndarray,
    actions: jnp.ndarray,
    dt: float = 0.1,
    action_len: int = 5,
    global_frame: bool = False,
    x_index: int = 0,
    y_index: int = 1,
    vel_x_index: int = 3,
    vel_y_index: int = 4,
    yaw_index: int = 5,
):
    """
    Roll out kinematic trajectories from current state and piecewise-constant actions.

    Args:
        current_states: Tensor with trailing shape [..., F]. Defaults assume
            repo feature order [x, y, z, vel_x, vel_y, yaw].
        actions: Tensor with trailing shape [..., K, 2] storing [accel, yaw_rate].
        global_frame: If True, integrate in the current global pose. If False,
            start from the origin with zero yaw.

    Returns:
        Tensor with shape [..., K * action_len, 5] storing [x, y, yaw, vel_x, vel_y].
    """
    current_states = jnp.asarray(current_states)
    actions = jnp.asarray(actions)

    x0 = current_states[..., x_index]
    y0 = current_states[..., y_index]
    theta0 = current_states[..., yaw_index]
    v_x0 = current_states[..., vel_x_index]
    v_y0 = current_states[..., vel_y_index]
    v0 = jnp.sqrt(v_x0**2 + v_y0**2)

    accel = jnp.repeat(actions[..., 0], action_len, axis=-1)
    yaw_rate = jnp.repeat(actions[..., 1], action_len, axis=-1)

    v = jnp.maximum(v0[..., None] + jnp.cumsum(accel * dt, axis=-1), 0.0)
    if global_frame:
        theta = wrap_angle(theta0[..., None] + jnp.cumsum(yaw_rate * dt, axis=-1))
        x = x0[..., None] + jnp.cumsum(v * jnp.cos(theta) * dt, axis=-1)
        y = y0[..., None] + jnp.cumsum(v * jnp.sin(theta) * dt, axis=-1)
    else:
        theta = wrap_angle(jnp.cumsum(yaw_rate * dt, axis=-1))
        x = jnp.cumsum(v * jnp.cos(theta) * dt, axis=-1)
        y = jnp.cumsum(v * jnp.sin(theta) * dt, axis=-1)

    v_x = v * jnp.cos(theta)
    v_y = v * jnp.sin(theta)
    return jnp.stack([x, y, theta, v_x, v_y], axis=-1)
