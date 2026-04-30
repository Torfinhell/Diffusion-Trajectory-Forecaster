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
        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_xy.shape)
        y = diffusion_sampler.add_noise(gt_xy, noise, timestep)
        pred_xy = model(
            jnp.asarray(timestep, dtype=gt_xy.dtype),
            y,
            _prepare_conditioning(model, batch),
        )
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
