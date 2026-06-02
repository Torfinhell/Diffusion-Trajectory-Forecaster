import jax.numpy as jnp

from src.utils.trajectory_transform import wrap_angle


def inverse_kinematics(path, valid, action_len: int, dt: float = 0.1):
    assert (
        path.shape[-2] % action_len == 1
    )  # TODO maybe support for not divisable much harder
    if valid.ndim == path.ndim:
        valid = valid[..., 0]
    num_timesteps = path.shape[-2]
    num_actions = (num_timesteps - 1) // action_len

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
    x0 = current_state[..., 0]
    y0 = current_state[..., 1]
    theta0 = current_state[..., 2]
    v_x0 = current_state[..., 3]
    v_y0 = current_state[..., 4]
    speed = jnp.sqrt(v_x0**2 + v_y0**2)

    accel = jnp.repeat(actions[..., 0], action_len, axis=-1)
    yaw_rate = jnp.repeat(actions[..., 1], action_len, axis=-1)
    speed_t = jnp.maximum(speed[..., None] + jnp.cumsum(accel * dt, axis=-1), 0.0)

    if global_frame:
        theta = wrap_angle(theta0[..., None] + jnp.cumsum(yaw_rate * dt, axis=-1))
        v_x = speed_t * jnp.cos(theta)
        v_y = speed_t * jnp.sin(theta)
        x = x0[..., None] + jnp.cumsum(v_x * dt, axis=-1)
        y = y0[..., None] + jnp.cumsum(v_y * dt, axis=-1)
    else:
        theta = wrap_angle(jnp.cumsum(yaw_rate * dt, axis=-1))
        v_x = speed_t * jnp.cos(theta)
        v_y = speed_t * jnp.sin(theta)
        x = jnp.cumsum(v_x * dt, axis=-1)
        y = jnp.cumsum(v_y * dt, axis=-1)

    return jnp.stack([x, y, theta, v_x, v_y], axis=-1)
