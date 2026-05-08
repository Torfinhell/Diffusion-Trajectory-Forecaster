import jax
import jax.numpy as jnp


@jax.jit
def fde(
    pred_xy: jnp.ndarray,
    gt_xy: jnp.ndarray,
    agents_coeffs: jnp.ndarray,
    future_valid: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Final Displacement Error (FDE).

    For each agent, uses the *last valid* timestep (per `future_valid`).
    Supports any leading batch dims. Expected trailing dims:
    - pred_xy, gt_xy: (..., A, H, 2)
    - agents_coeffs: (..., A)
    - future_valid: (..., A, H, 1) or (..., A, H)
    """
    pred_xy = jnp.asarray(pred_xy)
    gt_xy = jnp.asarray(gt_xy)
    agents_coeffs = jnp.asarray(agents_coeffs, dtype=jnp.float32)
    future_valid = jnp.asarray(future_valid)
    if future_valid.ndim == pred_xy.ndim - 1:
        future_valid = future_valid[..., None]

    diff = pred_xy - gt_xy
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (..., A, H)

    valid = future_valid[..., 0].astype(bool)  # (..., A, H)
    time_idx = jnp.arange(valid.shape[-1], dtype=jnp.int32)
    last_valid_idx = jnp.max(jnp.where(valid, time_idx, -1), axis=-1)  # (..., A)
    has_valid = last_valid_idx >= 0
    last_valid_idx = jnp.maximum(last_valid_idx, 0)

    dist_last = jnp.take_along_axis(dist, last_valid_idx[..., None], axis=-1)[..., 0]
    weights = agents_coeffs * has_valid.astype(jnp.float32)
    sum_error = jnp.sum(dist_last * weights)
    count = jnp.sum(weights)
    return sum_error / (count + eps)
