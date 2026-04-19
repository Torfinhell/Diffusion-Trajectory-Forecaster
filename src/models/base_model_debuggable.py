import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.models.base_model import BaseDiffusionModel
from src.models.base_model_debug import (
    compute_one_step_denoise_ade,
    debug_denoiser_scale,
    debug_metric_sample,
    debug_training_shapes,
)
from src.models.base_model_eval import (
    batch_coord_scale,
    log_images,
    mask_pred_for_plot,
    metric_log_name,
    on_validation_epoch_end as run_validation_epoch_end,
    plot_vis_kwargs,
    to_metric_frame,
    to_world_frame,
)
from src.models.base_model_oracle import (
    compute_batch_loss as compute_oracle_batch_loss,
    oracle_enabled,
    oracle_sampling_mode,
    sampling_t0,
)
from src.models.base_model_proxy import compute_proxy_batch_loss


class DebuggableBaseDiffusionModel(BaseDiffusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_cfg = kwargs.get("oracle_cfg") or {}
        self._shape_debug_printed = False

    def _oracle_enabled(self, key):
        return oracle_enabled(self, key)

    def _oracle_sampling_mode(self):
        return oracle_sampling_mode(self)

    def _sampling_t0(self):
        return sampling_t0(self)

    def training_step(self, batch):
        with jax.profiler.StepTraceAnnotation("train", step_num=int(self.global_step_)):
            if not self._shape_debug_printed:
                debug_training_shapes(self, batch)
                self._shape_debug_printed = True
            if self._oracle_enabled("use_for_train_loss"):
                self.train_key, loss_key = jr.split(self.train_key)
                value = self.compute_batch_loss(batch, loss_key, use_oracle=True)
            else:
                value, self.model, self.train_key, self.opt_state = (
                    BaseDiffusionModel.make_step(
                        self.model,
                        self.batch_loss_fn,
                        self.weight,
                        self.int_beta,
                        self.prediction_target,
                        batch,
                        self.t1,
                        self.train_key,
                        self.opt_state,
                        self.optim.update,
                    )
                )
            self.train_loss_tracker.update("train_loss", jnp.asarray(value))
            self.global_step_ += 1
            if (
                self._should_run_metrics_this_epoch("train")
                and len(self._train_batches_for_metrics) < self.train_metric_batches
            ):
                self._train_batches_for_metrics.append(batch)
            return None

    def validation_step(self, batch):
        with jax.profiler.StepTraceAnnotation(
            "validation", step_num=int(self.current_epoch)
        ):
            self.loader_key, val_key = jr.split(self.loader_key)
            value = self.compute_batch_loss(
                batch,
                val_key,
                use_oracle=self._oracle_enabled("use_for_val_loss"),
            )
            self.val_loss_tracker.update("val_loss", jnp.asarray(value))
            if bool(self.proxy_val_cfg.get("enabled", False)):
                self.loader_key, proxy_key = jr.split(self.loader_key)
                proxy_value = compute_proxy_batch_loss(self, batch, proxy_key)
                self.val_loss_tracker.update("val_proxy_loss", jnp.asarray(proxy_value))
            if (
                self.metrics is not None
                and self._should_run_metrics_this_epoch("val")
                and len(self._val_batches_for_metrics) < self.val_metric_batches
            ):
                self._val_batches_for_metrics.append(batch)

    def compute_batch_loss(self, batch, key, use_oracle=False):
        return compute_oracle_batch_loss(self, batch, key, use_oracle=use_oracle)

    def _compute_one_step_denoise_ade(self, batch):
        return compute_one_step_denoise_ade(self, batch)

    def _debug_denoiser_scale(self, batch):
        return debug_denoiser_scale(self, batch)

    def _update_metrics_for_batch(
        self, metrics, batch, return_first_prediction=False
    ):
        first_pred_xy = None
        num_solutions = int(
            self.vis.get(
                "num_trajectory_samples",
                self.vis.get("sample_steps", 1),
            )
        )
        debug_metrics = bool(self.vis.get("debug_metrics", False))

        # The debug subclass keeps the same metric flow as the base class, but
        # can print per-sample diagnostics before the metric state is updated.
        for sample_idx in range(batch["gt_xy"].shape[0]):
            pred_xy = self.sample_multiple_sol(
                batch["context"][sample_idx],
                num_solutions=num_solutions,
                predict_shape=batch["gt_xy"][sample_idx].shape,
                oracle_gt_xy=(
                    batch["gt_xy"][sample_idx]
                    if self._oracle_enabled("use_for_sampling")
                    else None
                ),
            )

            coord_scale = batch_coord_scale(batch, sample_idx)
            pred_xy_metric = to_metric_frame(pred_xy, coord_scale)
            gt_xy_metric = to_metric_frame(batch["gt_xy"][sample_idx], coord_scale)

            if debug_metrics:
                gt_mask = batch["gt_xy_mask"][sample_idx][..., 0].astype(bool)
                debug_metric_sample(sample_idx, pred_xy_metric, gt_xy_metric, gt_mask)

            metrics.update(
                pred_xy_metric,
                gt_xy_metric,
                batch["gt_xy_mask"][sample_idx],
            )
            if return_first_prediction and sample_idx == 0:
                first_pred_xy = pred_xy_metric

        return first_pred_xy, None

    def on_train_epoch_end(self) -> None:
        train_losses = self.train_loss_tracker.result()
        if "train_loss" in train_losses:
            self.log(
                metric_log_name("train", "loss"),
                train_losses["train_loss"],
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        if not self._should_run_metrics_this_epoch("train"):
            self._train_batches_for_metrics.clear()
            return
        if len(self.metrics_train) == 0 or len(self._train_batches_for_metrics) == 0:
            return

        enable_train_visualization = bool(
            self.vis.get("enable_train_visualization", False)
        )
        debug_denoiser_scale_enabled = bool(
            self.vis.get("debug_denoiser_scale", False)
        )
        log_denoise_metric = bool(self.vis.get("log_denoise_metric", True))
        train_images = []
        vis_plot_kwargs = plot_vis_kwargs(self)
        denoise_metric_vals = []

        self.metrics_train.reset()
        for batch_idx, batch in enumerate(self._train_batches_for_metrics):
            if debug_denoiser_scale_enabled and batch_idx == 0:
                self._debug_denoiser_scale(batch)
            if log_denoise_metric and batch_idx == 0:
                denoise_metric_vals.append(self._compute_one_step_denoise_ade(batch))

            first_pred_xy, _ = self._update_metrics_for_batch(
                self.metrics_train,
                batch,
                return_first_prediction=enable_train_visualization and batch_idx == 0,
            )
            if (
                enable_train_visualization
                and batch_idx == 0
                and "scenario" in batch
                and first_pred_xy is not None
            ):
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
                            **vis_plot_kwargs,
                        )
                    )

        vals = self.metrics_train.compute()
        log_dict = {
            metric_log_name("train", k): float(jnp.asarray(v))
            for k, v in vals.items()
        }
        if denoise_metric_vals:
            log_dict[metric_log_name("train", "one_step_denoise_ade")] = float(
                sum(denoise_metric_vals) / len(denoise_metric_vals)
            )
        self.log_dict(log_dict, prog_bar=True)
        if enable_train_visualization and train_images:
            log_images(
                self,
                f"Train scenarios and predictions/epoch_{self.current_epoch}",
                train_images,
            )
        print(
            f"[Train metrics] epoch={self.current_epoch}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
        )
        self._train_batches_for_metrics.clear()

    def on_validation_epoch_end(self) -> None:
        self._save_local_checkpoint()
        return run_validation_epoch_end(self)

    def sample_multiple_sol(
        self,
        context,
        num_solutions=1,
        predict_shape=None,
        save_full=False,
        oracle_gt_xy=None,
        y1_override=None,
    ):
        if oracle_gt_xy is not None and self._oracle_sampling_mode() == "exact":
            return oracle_gt_xy
        if y1_override is None:
            return super().sample_multiple_sol(
                context,
                num_solutions=num_solutions,
                predict_shape=predict_shape,
                save_full=save_full,
                oracle_gt_xy=oracle_gt_xy,
            )

        # Debug runs sometimes need a fixed initial noise sample so repeated
        # evaluations are directly comparable.
        self.sample_key, key = jr.split(self.sample_key)
        y1_override = jnp.asarray(y1_override)
        if y1_override.ndim == len(predict_shape):
            y1_overrides = jnp.repeat(y1_override[None, ...], num_solutions, axis=0)
        else:
            y1_overrides = y1_override
            if y1_overrides.shape[0] != num_solutions:
                raise ValueError(
                    "y1_override must have shape predict_shape or "
                    f"({num_solutions}, *predict_shape), got {y1_overrides.shape}."
                )

        sample_keys = jr.split(key, num_solutions)
        pred_samples = jax.vmap(
            lambda kk, y1: self.sample_one_sol(
                self.model,
                self.int_beta,
                predict_shape,
                self.dt0,
                self.t1,
                context,
                save_full,
                oracle_gt_xy,
                y1,
                kk,
            )[0]
        )(sample_keys, y1_overrides)
        return jnp.mean(pred_samples, axis=0)

    def sample_one_sol(
        self,
        model,
        int_beta,
        data_shape,
        dt0,
        t1,
        context,
        save_full=False,
        oracle_gt_xy=None,
        y1_override=None,
        key=None,
    ):
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        t0 = self._sampling_t0()
        if y1_override is None:
            y1 = jr.normal(key, data_shape)
        else:
            y1 = jnp.asarray(y1_override, dtype=jnp.float32)
            if y1.shape != tuple(data_shape):
                raise ValueError(
                    f"y1_override shape {y1.shape} does not match data_shape {data_shape}."
                )
        num_steps = max(1, int(np.ceil((t1 - t0) / abs(dt0))))
        ts = jnp.linspace(t1, t0, num_steps + 1)
        x = y1
        path = []
        for step_idx in range(num_steps):
            t_cur = ts[step_idx]
            t_next = ts[step_idx + 1]
            alpha_cur, sigma_cur = self._alpha_sigma(int_beta, t_cur)
            alpha_next, sigma_next = self._alpha_sigma(int_beta, t_next)
            if oracle_gt_xy is not None:
                x0_pred = oracle_gt_xy
                eps_pred = (x - alpha_cur * x0_pred) / jnp.maximum(sigma_cur, 1e-5)
            else:
                pred = model(t_cur, x, context)
                if self.prediction_target == "x0":
                    x0_pred = pred
                    eps_pred = (x - alpha_cur * x0_pred) / jnp.maximum(
                        sigma_cur, 1e-5
                    )
                elif self.prediction_target == "epsilon":
                    eps_pred = pred
                    x0_pred = (x - sigma_cur * eps_pred) / jnp.maximum(
                        alpha_cur, 1e-5
                    )
                elif self.prediction_target == "score":
                    score_pred = pred
                    eps_pred = -sigma_cur * score_pred
                    x0_pred = (x + (sigma_cur**2) * score_pred) / jnp.maximum(
                        alpha_cur, 1e-5
                    )
                else:
                    raise ValueError(
                        f"Unsupported prediction_target '{self.prediction_target}'"
                    )
            x = alpha_next * x0_pred + sigma_next * eps_pred
            if save_full:
                path.append(x)
        if save_full:
            return jnp.stack(path, axis=0)
        return x[None, ...]
