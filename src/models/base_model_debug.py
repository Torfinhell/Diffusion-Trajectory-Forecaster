import jax.numpy as jnp
import jax.random as jr


def compute_one_step_denoise_ade(model, batch):
    sample_idx = 0
    t = jnp.array(min(float(model.t1) * 0.5, 1.0), dtype=jnp.float32)
    key = jr.PRNGKey(0)
    gt_xy = jnp.asarray(batch["gt_xy"][sample_idx])
    mean = gt_xy * jnp.exp(-0.5 * model.int_beta(t))
    var = jnp.maximum(1.0 - jnp.exp(-model.int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise = jr.normal(key, gt_xy.shape)
    y = mean + std * noise
    pred_raw = jnp.asarray(model.model(t, y, batch["context"][sample_idx]))
    pred = model._prediction_to_x0(pred_raw, y, t)
    gt_mask = jnp.asarray(batch["gt_xy_mask"][sample_idx])
    if pred.ndim + 1 == gt_xy.ndim and gt_xy.shape[0] == 1:
        gt_xy = jnp.squeeze(gt_xy, axis=0)
        gt_mask = jnp.squeeze(gt_mask, axis=0)
    ade_metric = model.metrics_train.metrics[0].__class__(name="tmp_ADE")
    ade_metric.update(pred, gt_xy, gt_mask)
    return float(jnp.asarray(ade_metric.compute()))


def debug_denoiser_scale(model, batch):
    sample_idx = 0
    t = jnp.array(min(float(model.t1) * 0.5, 1.0), dtype=jnp.float32)
    key = jr.PRNGKey(0)
    gt_xy = jnp.asarray(batch["gt_xy"][sample_idx])
    mean = gt_xy * jnp.exp(-0.5 * model.int_beta(t))
    var = jnp.maximum(1.0 - jnp.exp(-model.int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise = jr.normal(key, gt_xy.shape)
    y = mean + std * noise
    context = jnp.asarray(batch["context"][sample_idx])
    pred = jnp.asarray(model.model(t, y, context))
    target = jnp.asarray(model._prediction_to_target(gt_xy, noise, std))
    valid = jnp.asarray(batch["gt_xy_mask"][sample_idx])
    if gt_xy.ndim == pred.ndim + 1 and gt_xy.shape[0] == 1:
        gt_xy = jnp.squeeze(gt_xy, axis=0)
        target = jnp.squeeze(target, axis=0)
        valid = jnp.squeeze(valid, axis=0)
    if valid.shape == gt_xy.shape:
        valid = valid[..., 0]
    if valid.ndim == gt_xy.ndim and valid.shape[-1] == 1:
        valid = jnp.squeeze(valid, axis=-1)
    valid = valid.astype(bool)
    if pred.shape != gt_xy.shape:
        raise ValueError(f"Unexpected pred shape {pred.shape} for gt_xy {gt_xy.shape}")
    if valid.shape != gt_xy.shape[:-1]:
        raise ValueError(
            f"Unexpected gt_xy_mask shape {valid.shape} for gt_xy {gt_xy.shape}"
        )
    pred_valid = pred[valid]
    target_valid = target[valid]
    if pred_valid.size == 0:
        return
    print(
        "[Denoiser debug] "
        f"pred_abs_mean={float(jnp.mean(jnp.abs(pred_valid))):.3f} "
        f"target_abs_mean={float(jnp.mean(jnp.abs(target_valid))):.3f} "
        f"pred_std={float(jnp.std(pred_valid)):.3f} "
        f"target_std={float(jnp.std(target_valid)):.3f} "
        f"pred_min={float(jnp.min(pred_valid)):.3f} "
        f"pred_max={float(jnp.max(pred_valid)):.3f} "
        f"target_min={float(jnp.min(target_valid)):.3f} "
        f"target_max={float(jnp.max(target_valid)):.3f}"
    )


def debug_training_shapes(model, batch):
    sample_idx = 0
    gt_xy = jnp.asarray(batch["gt_xy"][sample_idx])
    context = jnp.asarray(batch["context"][sample_idx])
    gt_xy_mask = jnp.asarray(batch["gt_xy_mask"][sample_idx])

    t = jnp.array(min(float(model.t1) * 0.5, 1.0), dtype=jnp.float32)
    key = jr.PRNGKey(0)
    noise = jr.normal(key, gt_xy.shape)
    mean = gt_xy * jnp.exp(-0.5 * model.int_beta(t))
    std = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-model.int_beta(t)), 1e-5))
    y = mean + std * noise
    pred = jnp.asarray(model.model(t, y, context))

    aligned_gt_xy = gt_xy
    aligned_mask = gt_xy_mask
    if aligned_gt_xy.ndim == pred.ndim + 1 and aligned_gt_xy.shape[0] == 1:
        aligned_gt_xy = jnp.squeeze(aligned_gt_xy, axis=0)
    if aligned_mask.ndim == pred.ndim + 1 and aligned_mask.shape[0] == 1:
        aligned_mask = jnp.squeeze(aligned_mask, axis=0)
    if aligned_mask.shape == aligned_gt_xy.shape:
        aligned_mask = aligned_mask[..., 0]
    if aligned_mask.ndim == aligned_gt_xy.ndim and aligned_mask.shape[-1] == 1:
        aligned_mask = jnp.squeeze(aligned_mask, axis=-1)

    print(
        "[Shape debug] "
        f"batch.gt_xy={batch['gt_xy'].shape} "
        f"batch.context={batch['context'].shape} "
        f"batch.gt_xy_mask={batch['gt_xy_mask'].shape}"
    )
    print(
        "[Shape debug] "
        f"sample.gt_xy={gt_xy.shape} "
        f"sample.context={context.shape} "
        f"sample.gt_xy_mask={gt_xy_mask.shape} "
        f"noisy_input={y.shape} "
        f"model_pred={pred.shape}"
    )
    print(
        "[Shape debug] "
        f"aligned_gt_xy={aligned_gt_xy.shape} "
        f"aligned_mask={aligned_mask.shape} "
        f"pred_matches_gt={pred.shape == aligned_gt_xy.shape} "
        f"mask_matches_gt_time={aligned_mask.shape == aligned_gt_xy.shape[:-1]}"
    )


def debug_metric_sample(sample_idx, pred_xy_metric, gt_xy_metric, gt_mask):
    pred_valid = pred_xy_metric[gt_mask]
    gt_valid = gt_xy_metric[gt_mask]
    if pred_valid.size == 0:
        return
    diff = pred_valid - gt_valid
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    print(
        "[Metric debug] "
        f"sample={sample_idx} "
        f"pred_abs_mean={float(jnp.mean(jnp.abs(pred_valid))):.3f} "
        f"gt_abs_mean={float(jnp.mean(jnp.abs(gt_valid))):.3f} "
        f"pred_min={float(jnp.min(pred_valid)):.3f} "
        f"pred_max={float(jnp.max(pred_valid)):.3f} "
        f"gt_min={float(jnp.min(gt_valid)):.3f} "
        f"gt_max={float(jnp.max(gt_valid)):.3f} "
        f"dist_mean={float(jnp.mean(dist)):.3f} "
        f"dist_max={float(jnp.max(dist)):.3f}"
    )
