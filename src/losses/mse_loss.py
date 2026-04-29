import jax.numpy as jnp
import jax.random as jr


def _prepare_conditioning(model, batch):
    return model.prepare_conditioning(batch)


def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)


class MSELoss:
    """Weighted MSE diffusion training loss."""

    def __init__(self):
        super().__init__()

    def __call__(self, model, diffusion_sampler, batch, key):
        gt_xy = batch["agent_future"][..., :2]
        INPUT_DIM = gt_xy.shape
        mean = gt_xy * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, INPUT_DIM)
        y = mean + std * noise
        pred_xy = model(t, y, _prepare_conditioning(model, batch))
        err = (pred_xy - gt_xy) ** 2
        weights = jnp.asarray(batch["agents_coeffs"], dtype=err.dtype)[
            ..., None, None
        ] * jnp.asarray(batch["agent_future_valid"], dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        weighted_element_count = jnp.ones_like(err) * weights
        loss = (err * weights).sum() / jnp.maximum(weighted_element_count.sum(), 1.0)
        valid_weights = jnp.asarray(batch["agent_future_valid"], dtype=gt_xy.dtype)
        stats = {
            "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
            "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
            "valid_ratio": jnp.mean(valid_weights),
        }
        return loss, stats
