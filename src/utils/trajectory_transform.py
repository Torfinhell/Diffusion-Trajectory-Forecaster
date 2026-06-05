import jax.numpy as jnp


def wrap_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def transform_polylines_to_local(polylines: jnp.ndarray) -> jnp.ndarray:
    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]
    ref_x = x[..., 0, None]
    ref_y = y[..., 0, None]
    ref_theta = theta[..., 0, None]
    cos_t = jnp.cos(ref_theta)
    sin_t = jnp.sin(ref_theta)

    local_x = (x - ref_x) * cos_t + (y - ref_y) * sin_t
    local_y = -(x - ref_x) * sin_t + (y - ref_y) * cos_t
    local_theta = wrap_angle(theta - ref_theta)
    local_polylines = jnp.stack([local_x, local_y, local_theta], axis=-1)
    valid = jnp.any(polylines[..., :3] != 0, axis=-1, keepdims=True)
    local_polylines = jnp.where(valid, local_polylines, 0.0)
    if polylines.shape[-1] > 3:
        return jnp.concatenate([local_polylines, polylines[..., 3:]], axis=-1)
    return local_polylines
