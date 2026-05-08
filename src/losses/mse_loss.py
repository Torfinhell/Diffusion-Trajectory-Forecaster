import jax.numpy as jnp
import jax.random as jr

def masked_abs_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (jnp.abs(values) * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def masked_std(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    weight_sum = jnp.maximum(weights.sum(), 1.0)
    mean = (values * weights).sum() / weight_sum
    var = ((values - mean) ** 2 * weights).sum() / weight_sum
    return jnp.sqrt(jnp.maximum(var, 0.0))


def masked_mean(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    return (values * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def masked_var(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    weight_sum = jnp.maximum(weights.sum(), 1.0)
    mean = (values * weights).sum() / weight_sum
    return ((values - mean) ** 2 * weights).sum() / weight_sum


def masked_mean_coord(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    weight_sum = jnp.maximum(weights.sum(axis=(0, 1), keepdims=False), 1.0)
    return (values * weights).sum(axis=(0, 1), keepdims=False) / weight_sum


def masked_std_coord(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights, dtype=values.dtype)
    weight_sum = jnp.maximum(weights.sum(axis=(0, 1), keepdims=False), 1.0)
    mean = (values * weights).sum(axis=(0, 1), keepdims=False) / weight_sum
    var = (((values - mean[None, None, :]) ** 2) * weights).sum(
        axis=(0, 1), keepdims=False
    ) / weight_sum
    return jnp.sqrt(jnp.maximum(var, 0.0))


def ade_proxy(pred_xy, gt_xy, valid_weights):
    """Mean L2 distance over valid agents/timesteps (unnormalized, in local coords)."""
    dist = jnp.sqrt(jnp.sum((pred_xy - gt_xy) ** 2, axis=-1))  # (A, T)
    w = jnp.squeeze(valid_weights, axis=-1) if valid_weights.ndim == 3 else valid_weights
    return masked_mean(dist, w)


class MSELoss:
    """Weighted MSE diffusion training loss.

    Args:
        snr_weighting: If True, apply min-SNR-5 weighting (down-weights t≈0
            timesteps). Set False when overfitting to ensure equal coverage of
            all noise levels, especially t=0 which drives sampling quality.
    """

    def __init__(self, snr_weighting: bool = True):
        self.snr_weighting = snr_weighting

    def __call__(self, model, diffusion_sampler, batch, key):
        gt_xy = batch["agent_future"][..., :2]
        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        valid_weights = jnp.asarray(batch["agent_future_valid"], dtype=gt_xy.dtype)
        x0_mean = jnp.asarray(batch["x0_mean"], dtype=gt_xy.dtype)[None, None, :]
        x0_std = jnp.sqrt(
            jnp.maximum(jnp.asarray(batch["x0_var"], dtype=gt_xy.dtype), 1e-6)
        )[None, None, :]
        gt_xy_model = (gt_xy - x0_mean) / x0_std
        noise = jr.normal(noise_key, gt_xy.shape)
        y = diffusion_sampler.add_noise(gt_xy_model, noise, timestep)
        timestep_f = jnp.asarray(timestep, dtype=gt_xy.dtype) / jnp.maximum(diffusion_sampler.num_steps - 1, 1)
        pred_xy_model = model(
            timestep_f,
            y,
            batch,
        )
        err = (pred_xy_model - gt_xy_model) ** 2

        # min-SNR-5 weighting: prevents high-noise timesteps from dominating.
        # NOTE: also heavily down-weights t≈0 (low-noise) steps (weight≈5/SNR≈0).
        # Disable for overfitting/debugging to get uniform coverage.
        alpha_prod = diffusion_sampler.alphas_cumprod[timestep]
        snr = alpha_prod / jnp.maximum(1.0 - alpha_prod, 1e-8)
        snr_weight = jnp.where(
            self.snr_weighting,
            jnp.minimum(snr, 5.0) / jnp.maximum(snr, 1e-8),
            1.0,
        )

        weights = jnp.asarray(batch["agents_coeffs"], dtype=err.dtype)[
            ..., None, None
        ] * jnp.asarray(batch["agent_future_valid"], dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        weighted_element_count = jnp.ones_like(err) * weights
        loss = snr_weight * (err * weights).sum() / jnp.maximum(weighted_element_count.sum(), 1.0)
        pred_xy = pred_xy_model * x0_std + x0_mean

        # t=0 probe: pass gt_xy as x_t (no noise) — if model can overfit this, loss pathway works
        pred_at_t0_model = model(
            jnp.zeros((), dtype=gt_xy.dtype),
            gt_xy_model,
            batch,
        )
        pred_at_t0 = pred_at_t0_model * x0_std + x0_mean
        ade_at_t0 = ade_proxy(pred_at_t0, gt_xy, valid_weights)
        mse_at_t0 = masked_mean((pred_at_t0_model - gt_xy_model) ** 2, valid_weights)

        x0_norm_mean_xy = masked_mean_coord(gt_xy_model, valid_weights)
        x0_norm_std_xy = masked_std_coord(gt_xy_model, valid_weights)
        stats = {
            "noisy_abs_mean": masked_abs_mean(y, valid_weights),
            "target_abs_mean": masked_abs_mean(gt_xy, valid_weights),
            "pred_abs_mean": masked_abs_mean(pred_xy, valid_weights),
            "traj_local_mean": masked_mean(gt_xy, valid_weights),
            "traj_local_var": masked_var(gt_xy, valid_weights),
            "gt_xy_std": masked_std(gt_xy, valid_weights),
            "x_t_std": masked_std(y, valid_weights),
            "x_t_mean": masked_mean(y, valid_weights),
            "x0_norm_mean": masked_mean(gt_xy_model, valid_weights),
            "x0_norm_std": masked_std(gt_xy_model, valid_weights),
            "x0_norm_mean_x": x0_norm_mean_xy[0],
            "x0_norm_mean_y": x0_norm_mean_xy[1],
            "x0_norm_std_x": x0_norm_std_xy[0],
            "x0_norm_std_y": x0_norm_std_xy[1],
            "valid_ratio": jnp.mean(valid_weights),
            "snr_weight": snr_weight,
            "timestep": jnp.asarray(timestep, dtype=gt_xy.dtype),
            # overfit diagnostics
            "ade_at_t0": ade_at_t0,
            "mse_at_t0": mse_at_t0,
        }
        return loss, stats
