import jax.numpy as jnp



def wrap_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


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
    if squeeze_batch:
        return transformed[0]
    return transformed

