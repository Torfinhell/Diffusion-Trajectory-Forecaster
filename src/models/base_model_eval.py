import jax.numpy as jnp
from src.visualization.viz import plot_simulator_state


def plot_vis_kwargs(model):
    excluded = {
        "debug_metrics",
        "debug_denoiser_scale",
        "num_trajectory_samples",
        "sample_steps",
        "sample_video_fps",
        "enable_visualization",
        "enable_train_visualization",
    }
    return {k: v for k, v in model.vis.items() if k not in excluded}


def to_world_frame(pred_xy, origin_xy):
    return pred_xy + origin_xy


def to_metric_frame(pred_xy, coord_scale):
    return pred_xy * coord_scale


def log_images(model, key, images):
    logger = getattr(model, "logger", None)
    if logger is None or not hasattr(logger, "log_image"):
        return
    logger.log_image(key=key, images=images, step=int(model.global_step_))


def log_video(model, key, path):
    logger = getattr(model, "logger", None)
    if logger is None or not hasattr(logger, "log_video"):
        return
    logger.log_video(key=key, path=path, step=int(model.global_step_))


def metric_log_name(split, name):
    return f"{split}_{name}"


def mask_pred_for_plot(pred_xy, gt_xy_mask):
    pred_xy = jnp.asarray(pred_xy)
    gt_xy_mask = jnp.asarray(gt_xy_mask)
    # Metrics can use batched or unbatched masks; plotting only needs the shape
    # to line up with the predicted trajectory being drawn.
    while gt_xy_mask.ndim > pred_xy.ndim:
        gt_xy_mask = jnp.squeeze(gt_xy_mask, axis=0)
    while gt_xy_mask.ndim < pred_xy.ndim:
        gt_xy_mask = gt_xy_mask[None, ...]
    return jnp.where(gt_xy_mask.astype(bool), pred_xy, jnp.nan)


def batch_coord_scale(batch, sample_idx):
    coord_scale = batch.get("coord_scale")
    if coord_scale is None:
        return jnp.array([1.0], dtype=jnp.float32)
    return coord_scale[sample_idx]


def update_metrics_for_batch(model, metrics, batch, return_first_prediction=False):
    first_pred_xy = None
    num_solutions = model._num_metric_trajectory_samples()

    for sample_idx in range(batch["gt_xy"].shape[0]):
        # Sampling stays in the model's normalized frame; metrics are computed
        # after restoring the per-scene coordinate scale.
        pred_xy = model.sample_multiple_sol(
            batch["context"][sample_idx],
            num_solutions=num_solutions,
            predict_shape=batch["gt_xy"][sample_idx].shape,
            oracle_gt_xy=(
                batch["gt_xy"][sample_idx]
                if model._oracle_enabled("use_for_sampling")
                else None
            ),
        )
        coord_scale = batch_coord_scale(batch, sample_idx)
        pred_xy_metric = to_metric_frame(pred_xy, coord_scale)
        gt_xy_metric = to_metric_frame(batch["gt_xy"][sample_idx], coord_scale)
        metrics.update(
            pred_xy_metric,
            gt_xy_metric,
            batch["gt_xy_mask"][sample_idx],
        )
        if return_first_prediction and sample_idx == 0:
            first_pred_xy = pred_xy_metric

    return first_pred_xy, None


def on_train_epoch_end(model):
    train_losses = model.train_loss_tracker.result()
    if "train_loss" in train_losses:
        model.log(
            metric_log_name("train", "loss"),
            train_losses["train_loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
    if not model._should_run_metrics_this_epoch("train"):
        model._train_batches_for_metrics.clear()
        return
    if len(model.metrics_train) == 0 or len(model._train_batches_for_metrics) == 0:
        return

    enable_train_visualization = bool(model.vis.get("enable_train_visualization", False))
    train_images = []
    plot_kwargs = plot_vis_kwargs(model)

    model.metrics_train.reset()
    for batch_idx, batch in enumerate(model._train_batches_for_metrics):
        first_pred_xy, _ = model._update_metrics_for_batch(
            model.metrics_train,
            batch,
            return_first_prediction=enable_train_visualization and batch_idx == 0,
        )
        if (
            enable_train_visualization
            and batch_idx == 0
            and "scenario" in batch
            and first_pred_xy is not None
        ):
            # Only the first cached batch is rendered to keep epoch-end logging
            # cheap while still giving a visual sanity check.
            viz_state = batch["scenario"]
            if viz_state is not None:
                first_pred_xy_plot = mask_pred_for_plot(
                    first_pred_xy, batch["gt_xy_mask"][0]
                )
                first_pred_xy_world = to_world_frame(
                    first_pred_xy_plot,
                    batch["origin_xy"][0],
                )
                train_images.append(
                    plot_simulator_state(
                        viz_state,
                        batch_idx=0,
                        pred_xy=first_pred_xy_world,
                        **plot_kwargs,
                    )
                )

    vals = model.metrics_train.compute()
    log_dict = {metric_log_name("train", k): float(jnp.asarray(v)) for k, v in vals.items()}
    model.log_dict(log_dict, prog_bar=True)
    if enable_train_visualization and train_images:
        log_images(
            model,
            f"Train scenarios and predictions/epoch_{model.current_epoch}",
            train_images,
        )
    print(
        f"[Train metrics] epoch={model.current_epoch}  "
        + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
    )
    model._train_batches_for_metrics.clear()


def on_validation_epoch_end(model):
    val_losses = model.val_loss_tracker.result()
    if "val_loss" in val_losses:
        model.log(
            metric_log_name("val", "loss"),
            val_losses["val_loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
    if model._proxy_val_enabled() and "val_proxy_loss" in val_losses:
        model.log(
            metric_log_name("val", "proxy_loss"),
            val_losses["val_proxy_loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
    if not model._should_run_metrics_this_epoch("val"):
        model._val_batches_for_metrics.clear()
        return
    if len(model.metrics_val) == 0 or len(model._val_batches_for_metrics) == 0:
        return

    enable_visualization = bool(model.vis.get("enable_visualization", False))
    images = []
    plot_kwargs = plot_vis_kwargs(model)

    model.metrics_val.reset()
    for batch_idx, batch in enumerate(model._val_batches_for_metrics):
        first_pred_xy, _ = model._update_metrics_for_batch(
            model.metrics_val,
            batch,
            return_first_prediction=batch_idx == 0,
        )
        if (
            enable_visualization
            and batch_idx == 0
            and "scenario" in batch
            and first_pred_xy is not None
        ):
            # Validation uses the same single-scene rendering rule as training so
            # visualization cost does not scale with the number of cached batches.
            viz_state = batch["scenario"]
            if viz_state is not None:
                first_pred_xy_plot = mask_pred_for_plot(
                    first_pred_xy, batch["gt_xy_mask"][0]
                )
                first_pred_xy_world = to_world_frame(
                    first_pred_xy_plot, batch["origin_xy"][0]
                )
                images.append(
                    model._render_prediction_image(
                        viz_state, first_pred_xy_world, plot_kwargs
                    )
                )

    if enable_visualization and images:
        log_images(
            model,
            f"Scenarios and predictions/epoch_{model.current_epoch}",
            images,
        )

    vals = model.metrics_val.compute()
    log_dict = {metric_log_name("val", k): float(jnp.asarray(v)) for k, v in vals.items()}
    model.log_dict(log_dict, prog_bar=True)
    model._val_batches_for_metrics.clear()
