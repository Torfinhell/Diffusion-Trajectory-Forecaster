import jax
import jax.numpy as jnp

def wrap_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jax.jit(static_argnames=["x_index", "y_index", "heading_index"])
def batch_transform_trajs_to_local_frame(
    trajs,
    origin_xy,
    origin_theta,
    x_index=0,
    y_index=1,
    heading_index=5,
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

    x = trajs[..., x_index] # (B, num_agents, num_timesteps)
    y = trajs[..., y_index]
    theta = trajs[..., heading_index]
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
    local_theta = wrap_angle(theta - origin_theta)

    transformed = trajs.at[..., x_index].set(local_x)
    transformed = transformed.at[..., y_index].set(local_y)
    transformed = transformed.at[..., heading_index].set(local_theta)

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

@jax.jit(static_argnames=["x_index", "y_index", "heading_index"])
def batch_transform_trajs_to_global_frame(
    trajs,
    origin_xy,
    origin_theta,
    x_index=0,
    y_index=1,
    heading_index=None,
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
