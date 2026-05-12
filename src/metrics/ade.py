import jax
import jax.numpy as jnp


@jax.jit
def ade(
    pred_xy: jnp.ndarray,
    gt_xy: jnp.ndarray,
    agents_valid: jnp.ndarray,
    future_valid: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Average Displacement Error (ADE).

    Supports any leading batch dims. Expected trailing dims:
    - pred_xy, gt_xy: (..., A, H, 2)
    - agents_valid: (..., A)
    - future_valid: (..., A, H, 1) or (..., A, H)
    """
    pred_xy = jnp.asarray(pred_xy)
    gt_xy = jnp.asarray(gt_xy)
    agents_valid = jnp.asarray(agents_valid, dtype=jnp.float32)
    future_valid = jnp.asarray(future_valid, dtype=jnp.float32)
    if future_valid.ndim == pred_xy.ndim - 1:
        future_valid = future_valid[..., None]

    diff = pred_xy - gt_xy
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (..., A, H)
    weights = agents_valid[..., None] * future_valid[..., 0]  # (..., A, H)
    sum_error = jnp.sum(dist * weights)
    count = jnp.sum(weights)
    return sum_error / (count + eps)
