import jax.numpy as jnp

def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def batch_target_stats(batch):
    gt_xy = jnp.asarray(batch["agent_future"][..., :2], dtype=jnp.float32)
    valid_weights = jnp.asarray(batch["agent_future_valid"], dtype=gt_xy.dtype)
    return {
        "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
        "valid_ratio": jnp.mean(
            jnp.asarray(batch["agent_future_valid"], dtype=gt_xy.dtype)
        ),
    }
