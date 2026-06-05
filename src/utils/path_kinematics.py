import jax.numpy as jnp

from src.utils.trajectory_transform import wrap_angle


def inverse_kinematics(path, valid, action_len: int, dt: float = 0.1):
    if valid.ndim == path.ndim:
        valid = valid[..., 0]
    num_timesteps = path.shape[-2]
    num_actions = (num_timesteps - 1) // action_len
    # truncate to action_len * num_actions + 1 timesteps so diff gives N*action_len values
    keep = num_actions * action_len + 1
    path = path[..., :keep, :]
    valid = valid[..., :keep]

    yaw = path[..., 2]
    speed = jnp.sqrt(path[..., 3] ** 2 + path[..., 4] ** 2)
    yaw_rate = wrap_angle(jnp.diff(yaw, axis=-1)) / dt
    accel = jnp.diff(speed, axis=-1) / dt
    action_valid = valid[..., :-1] & valid[..., 1:]
    yaw_rate = jnp.where(action_valid, yaw_rate, 0.0)
    accel = jnp.where(action_valid, accel, 0.0)

    prefix = path.shape[:-2]
    yaw_rate = yaw_rate.reshape(*prefix, num_actions, action_len)
    accel = accel.reshape(*prefix, num_actions, action_len)
    action_valid = action_valid.reshape(*prefix, num_actions, action_len)

    valid_count = jnp.maximum(action_valid.sum(axis=-1), 1)
    actions = jnp.stack(
        [accel.sum(axis=-1) / valid_count, yaw_rate.sum(axis=-1) / valid_count],
        axis=-1,
    )
    actions_valid = jnp.any(action_valid, axis=-1)
    actions = jnp.where(actions_valid[..., None], actions, 0.0)
    return actions, actions_valid


def roll_out(
    current_state,
    actions,
    action_len: int,
    dt: float = 0.1,
    global_frame: bool = False,
):
    # Extract initial state components
    x0 = current_state[..., 0]
    y0 = current_state[..., 1]
    theta0 = current_state[..., 2]
    v_x0 = current_state[..., 3]
    v_y0 = current_state[..., 4]
    speed = jnp.sqrt(v_x0**2 + v_y0**2)

    # Repeat actions across time steps
    accel = jnp.repeat(actions[..., 0], action_len, axis=-1)
    yaw_rate = jnp.repeat(actions[..., 1], action_len, axis=-1)
    speed_t = jnp.maximum(speed[..., None] + jnp.cumsum(accel * dt, axis=-1), 0.0)

    if global_frame:
        theta_traj = wrap_angle(theta0[..., None] + jnp.cumsum(yaw_rate * dt, axis=-1))
        v_x_traj = speed_t * jnp.cos(theta_traj)
        v_y_traj = speed_t * jnp.sin(theta_traj)
        x_traj = x0[..., None] + jnp.cumsum(v_x_traj * dt, axis=-1)
        y_traj = y0[..., None] + jnp.cumsum(v_y_traj * dt, axis=-1)

        # Construct the initial state to prepend
        init_state = current_state[..., None, :]
    else:
        theta_traj = wrap_angle(jnp.cumsum(yaw_rate * dt, axis=-1))
        v_x_traj = speed_t * jnp.cos(theta_traj)
        v_y_traj = speed_t * jnp.sin(theta_traj)
        x_traj = jnp.cumsum(v_x_traj * dt, axis=-1)
        y_traj = jnp.cumsum(v_y_traj * dt, axis=-1)

        # Compute local initial velocities for the prepended state
        cos_t0, sin_t0 = jnp.cos(theta0), jnp.sin(theta0)
        local_v_x0 = v_x0 * cos_t0 + v_y0 * sin_t0
        local_v_y0 = -v_x0 * sin_t0 + v_y0 * cos_t0

        # Local frame sets initial x, y, and theta to 0.0
        zeros = jnp.zeros_like(local_v_x0)
        init_state = jnp.stack([zeros, zeros, zeros, local_v_x0, local_v_y0], axis=-1)[
            ..., None, :
        ]

    # Concatenate the initial state with the generated trajectory along the time dimension
    traj = jnp.stack([x_traj, y_traj, theta_traj, v_x_traj, v_y_traj], axis=-1)
    return jnp.concatenate([init_state, traj], axis=-2)
