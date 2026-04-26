import jax.numpy as jnp

from src.utils import maybe_save_best_checkpoint
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
    pred_xy = jnp.asarray(pred_xy)
    origin_xy = jnp.asarray(origin_xy)
    while origin_xy.ndim < pred_xy.ndim:
        origin_xy = origin_xy[..., None, :]
    return pred_xy + origin_xy


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
    return f"{split}/{name}"


def image_log_name(split, name):
    return f"images/{split}_{name}"


def mask_pred_for_plot(pred_xy, agents_coeffs):
    pred_xy = jnp.asarray(pred_xy)
    agents_coeffs = jnp.asarray(agents_coeffs)
    while agents_coeffs.ndim > pred_xy.ndim - 2:
        agents_coeffs = jnp.squeeze(agents_coeffs, axis=0)
    while agents_coeffs.ndim < pred_xy.ndim - 2:
        agents_coeffs = agents_coeffs[None, ...]
    valid_agents = (agents_coeffs > 0)[..., None, None]
    return jnp.where(valid_agents, pred_xy, jnp.nan)


def update_metrics_for_batch(model, metrics, batch, return_first_prediction=False):
    first_pred_xy = None
    num_solutions = int(
        model.vis.get(
            "num_trajectory_samples",
            model.vis.get("sample_steps", 1),
        )
    )

    for sample_idx in range(batch["agent_future"].shape[0]):
        gt_xy = batch["agent_future"][sample_idx][..., :2]
        future_valid = batch["agent_future_valid"][sample_idx]
        pred_xy = model.sample_multiple_sol(
            batch["agent_past"][sample_idx],
            num_solutions=num_solutions,
            predict_shape=gt_xy.shape,
        )
        metrics.update(
            pred_xy,
            gt_xy,
            batch["agents_coeffs"][sample_idx],
            future_valid,
        )
        if return_first_prediction and sample_idx == 0:
            first_pred_xy = pred_xy

    return first_pred_xy, None


def on_train_epoch_end(model):
    train_losses = {
        #otherwise it is ArrayImpl
        key: float(jnp.asarray(value))
        for key, value in model.train_loss_tracker.result().items()
    }
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

    enable_train_visualization = bool(
        model.vis.get("enable_train_visualization", False)
    )
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
                    first_pred_xy, batch["agents_coeffs"][0]
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
    log_dict = {
        metric_log_name("train", k): float(jnp.asarray(v)) for k, v in vals.items()
    }
    model.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
    if enable_train_visualization and train_images:
        log_images(
            model,
            image_log_name("train", "predictions"),
            train_images,
        )
    print(
        f"[Train metrics] epoch={model.current_epoch}  "
        + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
    )
    model._train_batches_for_metrics.clear()


def on_validation_epoch_end(model):
    checkpoint_metrics = {}
    val_losses = {
        #otherwise it is ArrayImpl
        key: float(jnp.asarray(value))
        for key, value in model.val_loss_tracker.result().items()
    }
    if "val_loss" in val_losses:
        model.log(
            metric_log_name("val", "loss"),
            val_losses["val_loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
    checkpoint_metrics.update(val_losses)
    if (
        bool(model.proxy_val_cfg.get("enabled", False))
        and "val_proxy_loss" in val_losses
    ):
        model.log(
            metric_log_name("val", "proxy_loss"),
            val_losses["val_proxy_loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
    if not model._should_run_metrics_this_epoch("val"):
        maybe_save_best_checkpoint(model, checkpoint_metrics)
        model._val_batches_for_metrics.clear()
        return
    if len(model.metrics_val) == 0 or len(model._val_batches_for_metrics) == 0:
        maybe_save_best_checkpoint(model, checkpoint_metrics)
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
                    first_pred_xy, batch["agents_coeffs"][0]
                )
                first_pred_xy_world = to_world_frame(
                    first_pred_xy_plot, batch["origin_xy"][0]
                )
                images.append(
                    plot_simulator_state(
                        viz_state,
                        batch_idx=0,
                        pred_xy=first_pred_xy_world,
                        **plot_kwargs,
                    )
                )

    if enable_visualization and images:
        log_images(
            model,
            image_log_name("val", "predictions"),
            images,
        )

    vals = model.metrics_val.compute()
    log_dict = {
        metric_log_name("val", k): float(jnp.asarray(v)) for k, v in vals.items()
    }
    model.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
    checkpoint_metrics.update(log_dict)
    maybe_save_best_checkpoint(model, checkpoint_metrics)
    model._val_batches_for_metrics.clear()


def on_test_epoch_end(model):
    if len(model.metrics_test) == 0:
        return

    vals = model.metrics_test.compute()
    log_dict = {
        metric_log_name("test", k): float(jnp.asarray(v)) for k, v in vals.items()
    }
    model.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
