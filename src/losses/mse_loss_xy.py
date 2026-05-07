import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)


class MSELossXY(eqx.Module):
    """Weighted MSE diffusion training loss."""

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        model,
        diffusion_sampler,
        agent_future,
        agents_coeffs,
        agent_future_valid,
        key,
        debug=False,
        **kwargs,
    ):
        gt_xy = agent_future[
            ..., :2
        ]  # TODO predict not x y but maybe all together some stuff
        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_xy.shape)
        y = diffusion_sampler.add_noise(gt_xy, noise, timestep)
        timestep_f = jnp.asarray(timestep, dtype=gt_xy.dtype) / jnp.maximum(
            diffusion_sampler.num_steps - 1, 1
        )
        pred_xy = model(
            timestep_f,
            y,
            **kwargs,
        )
        err = (pred_xy - gt_xy) ** 2
        weights = jnp.asarray(agents_coeffs, dtype=err.dtype)[
            ..., None, None
        ] * jnp.asarray(agent_future_valid, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        weighted_element_count = jnp.ones_like(err) * weights
        loss = (err * weights).sum() / jnp.maximum(weighted_element_count.sum(), 1.0)
        valid_weights = jnp.asarray(agent_future_valid, dtype=gt_xy.dtype)
        if not debug:
            return loss
        stats = {
            "noisy_abs_mean": masked_abs_mean(y, valid_weights),
            "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
            "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
            "valid_ratio": jnp.mean(valid_weights),
        }
        return loss, stats
