import tempfile
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from PIL import Image

try:
    from mocked_model import OracleDiffusionModel
except ImportError:
    OracleDiffusionModel = None
from src.data_module import data_process_scenarios
from src.metrics import MetricCollection
from src.visualization.viz import plot_simulator_state


class BaseDiffusionModel(L.LightningModule):
    CHECKPOINT_DIR = Path("checkpoints/ScoreBased")
    CHECKPOINT_PATH = CHECKPOINT_DIR / "last.eqx"

    def __init__(
        self,
        seed,
        load_last_checkpoint,
        cfg_metrics,
        grad_clip,
        vis_cfg,
        trainer_cfg=None,
        oracle_cfg=None,
        prediction_target="x0",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.key = jax.random.PRNGKey(seed)
        self.key, key_model, self.train_key, self.loader_key, self.sample_key = (
            jax.random.split(self.key, 5)
        )
        self.metrics = cfg_metrics
        self.vis = vis_cfg
        self.trainer_cfg = trainer_cfg or {}
        self.grad_clip = grad_clip
        self.load_last_checkpoint = load_last_checkpoint
        self.oracle_cfg = oracle_cfg or {}
        self.samples = 10
        self.global_step_ = 0  # TODO
        self.metrics_train = MetricCollection(
            [instantiate(m) for m in self.metrics.train]
        )
        self.metrics_val = MetricCollection([instantiate(m) for m in self.metrics.val])
        self.val_metric_batches = int(self.trainer_cfg.get("val_metric_batches", 3))
        self._val_batches_for_metrics = []
        self.train_metric_batches = int(
            self.trainer_cfg.get("train_metric_batches", 3)
        )
        self._train_batches_for_metrics = []
        self.metrics_prefix = "Val"
        self._shape_debug_printed = False
        self.prediction_target = str(prediction_target).lower()
        if self.prediction_target not in {"x0", "epsilon", "score"}:
            raise ValueError(
                f"Unsupported prediction_target '{prediction_target}'. "
                "Use one of: x0, epsilon, score."
            )
        self.model = self.get_model(key_model)
        self.configure_ddpm_scheduler()
        self.configure_optimizers()
        self.data_shape = None

    def get_model(self, cfg_model, key_model):
        raise NotImplementedError(
            "Should not use base class. Should implement get_model for child class"
        )

    def configure_ddpm_scheduler(self):
        raise NotImplementedError(
            "Should not use base class. Should implement configure_ddpm_scheduler for child class"
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "Should not use base class. Should implement configure_optimizers for child class"
        )

    def build_learning_rate(self, base_lr):
        scheduler_cfg = getattr(self, "lr_scheduler_cfg", None)
        if scheduler_cfg is None:
            return float(base_lr)

        name = str(scheduler_cfg.get("name", "none")).lower()
        if name in {"none", "off", "disabled"}:
            return float(base_lr)

        decay_steps = int(scheduler_cfg.get("decay_steps", 0))
        if decay_steps <= 0:
            raise ValueError(
                f"lr_scheduler.decay_steps must be > 0 for scheduler '{name}'"
            )

        end_lr = float(scheduler_cfg.get("end_lr", 0.0))

        if name == "cosine":
            alpha = end_lr / float(base_lr) if base_lr > 0 else 0.0
            return optax.cosine_decay_schedule(
                init_value=float(base_lr),
                decay_steps=decay_steps,
                alpha=alpha,
            )

        if name == "linear":
            return optax.linear_schedule(
                init_value=float(base_lr),
                end_value=end_lr,
                transition_steps=decay_steps,
            )

        raise ValueError(f"Unsupported lr scheduler '{name}'")

    def training_step(self, batch):
        if not self._shape_debug_printed:
            self._debug_training_shapes(batch)
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
        loss_value = jnp.asarray(value).item()
        self.log(
            "Train_Loss",
            loss_value,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        dict_ = {"loss": torch.scalar_tensor(loss_value)}
        self.global_step_ += 1
        if (
            self._should_run_metrics_this_epoch("train")
            and len(self._train_batches_for_metrics) < self.train_metric_batches
        ):
            self._train_batches_for_metrics.append(batch)
        return dict_

    def validation_step(self, batch):
        self.loader_key, val_key = jr.split(self.loader_key)
        value = self.compute_batch_loss(
            batch,
            val_key,
            use_oracle=self._oracle_enabled("use_for_val_loss"),
        )
        self.log(
            "Val_Loss",
            jnp.asarray(value).item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        # collect some first batches to compute metrics on
        if (
            self.metrics is not None
            and self._should_run_metrics_this_epoch("val")
            and len(self._val_batches_for_metrics) < self.val_metric_batches
        ):
            self._val_batches_for_metrics.append(batch)

    @staticmethod
    @eqx.filter_jit
    def make_step(
        model,
        batch_loss_fn,
        weight,
        int_beta,
        prediction_target,
        batch,
        t1,
        key,
        opt_state,
        opt_update,
    ):
        loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
        loss, grads = loss_fn(
            model, weight, int_beta, prediction_target, batch, t1, key
        )
        updates, opt_state = opt_update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        key = jr.split(key, 1)[0]
        return loss, model, key, opt_state

    def _oracle_enabled(self, key):
        return bool(self.oracle_cfg is not None and self.oracle_cfg.get(key, False))

    def _make_oracle_model(self, gt_xy):
        if OracleDiffusionModel is None:
            raise RuntimeError(
                "Oracle mode requested but mocked_model.py is not available."
            )
        return OracleDiffusionModel(gt_xy=gt_xy, int_beta=self.int_beta)

    def _oracle_sampling_mode(self):
        if self.oracle_cfg is None:
            return "exact"
        return self.oracle_cfg.get("sampling_mode", "exact")

    def _sampling_t0(self):
        if hasattr(self, "t0"):
            return float(self.t0)
        if self.oracle_cfg is None:
            return 1e-3
        return float(self.oracle_cfg.get("sampling_t0", 1e-3))

    def _plot_vis_kwargs(self):
        excluded = {
            "debug_metrics",
            "debug_denoiser_scale",
            "num_trajectory_samples",
            "sample_steps",
            "sample_video_fps",
            "enable_visualization",
            "enable_train_visualization",
            "direct_prediction_eval",
        }
        return {k: v for k, v in self.vis.items() if k not in excluded}

    @staticmethod
    def _to_world_frame(pred_xy, origin_xy):
        return pred_xy + origin_xy

    @staticmethod
    def _to_metric_frame(pred_xy, coord_scale):
        return pred_xy * coord_scale

    def _log_images(self, key, images):
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "log_image"):
            return
        logger.log_image(key=key, images=images, step=int(self.global_step))

    def _log_video(self, key, path):
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "log_video"):
            return
        logger.log_video(key=key, path=path, step=int(self.global_step))

    def _metric_every_n_epochs(self, split):
        key = f"{split}_metric_every_n_epochs"
        return max(1, int(self.trainer_cfg.get(key, 1)))

    def _should_run_metrics_this_epoch(self, split):
        every_n = self._metric_every_n_epochs(split)
        return (int(self.current_epoch) + 1) % every_n == 0

    def _num_metric_trajectory_samples(self):
        return int(
            self.vis.get(
                "num_trajectory_samples",
                self.vis.get("sample_steps", 1),
            )
        )

    def _upload_artifact(self, name, path, metadata=None):
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "upload_artifact"):
            return
        logger.upload_artifact(name=name, path=path, metadata=metadata)

    @staticmethod
    def _mask_pred_for_plot(pred_xy, gt_xy_mask):
        pred_xy = jnp.asarray(pred_xy)
        gt_xy_mask = jnp.asarray(gt_xy_mask)
        while gt_xy_mask.ndim > pred_xy.ndim:
            gt_xy_mask = jnp.squeeze(gt_xy_mask, axis=0)
        while gt_xy_mask.ndim < pred_xy.ndim:
            gt_xy_mask = gt_xy_mask[None, ...]
        return jnp.where(gt_xy_mask.astype(bool), pred_xy, jnp.nan)

    @staticmethod
    def _batch_coord_scale(batch, sample_idx):
        coord_scale = batch.get("coord_scale")
        if coord_scale is None:
            return jnp.array([1.0], dtype=jnp.float32)
        return coord_scale[sample_idx]

    def _direct_predict(self, context, predict_shape, num_solutions=1):
        t = jnp.array(min(float(self.t1) * 0.5, 1.0), dtype=jnp.float32)
        y = jnp.zeros(predict_shape, dtype=jnp.float32)
        pred_raw = jnp.asarray(self.model(t, y, context))
        pred_xy = self._prediction_to_x0(pred_raw, y, t)
        pred_path = jnp.repeat(pred_xy[None, ...], num_solutions, axis=0)
        return pred_xy, pred_path

    @staticmethod
    def _alpha_sigma(int_beta, t):
        alpha = jnp.exp(-0.5 * int_beta(t))
        sigma = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5))
        return alpha, sigma

    def _prediction_to_target(self, gt_xy, noise, std):
        if self.prediction_target == "x0":
            return gt_xy
        if self.prediction_target == "epsilon":
            return noise
        if self.prediction_target == "score":
            return -noise / jnp.maximum(std, 1e-5)
        raise ValueError(f"Unsupported prediction_target '{self.prediction_target}'")

    def _prediction_to_x0(self, pred, x_t, t):
        alpha, sigma = self._alpha_sigma(self.int_beta, t)
        if self.prediction_target == "x0":
            return pred
        if self.prediction_target == "epsilon":
            return (x_t - sigma * pred) / jnp.maximum(alpha, 1e-5)
        if self.prediction_target == "score":
            return (x_t + (sigma**2) * pred) / jnp.maximum(alpha, 1e-5)
        raise ValueError(f"Unsupported prediction_target '{self.prediction_target}'")

    def compute_batch_loss(self, batch, key, use_oracle=False):
        if not use_oracle:
            return self.batch_loss_fn(
                self.model,
                self.weight,
                self.int_beta,
                self.prediction_target,
                batch,
                self.t1,
                key,
            )

        batch_size = batch["gt_xy"].shape[0]
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        t = jr.uniform(tkey, (batch_size,), minval=0, maxval=self.t1 / batch_size)
        t = t + (self.t1 / batch_size) * jnp.arange(batch_size)

        losses = []
        for sample_idx in range(batch_size):
            sample_batch = {
                "gt_xy": batch["gt_xy"][sample_idx],
                "gt_xy_mask": batch["gt_xy_mask"][sample_idx],
                "context": batch["context"][sample_idx],
            }
            oracle_model = self._make_oracle_model(sample_batch["gt_xy"])
            losses.append(
                self.single_loss_fn(
                    oracle_model,
                    self.weight,
                    self.int_beta,
                    self.prediction_target,
                    sample_batch,
                    t[sample_idx],
                    losskey[sample_idx],
                )
            )
        return jnp.mean(jnp.stack(losses))

    @staticmethod
    def batch_loss_fn(model, weight, int_beta, prediction_target, batch, t1, key):

        batch_size = batch["gt_xy"].shape[0]
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        """
		Low-discrepancy sampling over t to reduce variance
		by sampling very evenly by sampling uniformly and independently from (t1-t0)/batch_size bins
		t = [U(0,1), U(1,2), U(2,3), ...]
		"""
        # t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
        # t = t + (t1 / batch_size) * jnp.arange(batch_size)
        t_min = min(0.1, float(t1) * 0.5)
        t = jr.uniform(tkey, (batch_size,), minval=t_min, maxval=t1)
        """ Fixing the first three arguments of single_loss_fn, leaving batch, t and key as input """
        loss_fn = partial(
            BaseDiffusionModel.single_loss_fn,
            model,
            weight,
            int_beta,
            prediction_target,
        )
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(batch, t, losskey))

    @staticmethod
    def single_loss_fn(model, weight, int_beta, prediction_target, batch, t, key):
        """
        OU process provides analytical mean and variance
        int_beta(t) = ß = θ
        E[X_t] = μ + exp[-θ t] ( X_0 - μ) w/ μ=0 gives =X_0 * exp[ - θ t ]
        V[X_t] = σ^2/(2θ) ( 1 - exp(-2 θ t) ) w/ σ^2=ß=θ gives = 1 - exp(-2 ß t)
        :param model:
        :param weight:
        :param int_beta:
        :param batch:
        :param t:
        :param key:
        :return:
        """
        INPUT_DIM = batch["gt_xy"].shape
        mean = batch["gt_xy"] * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, INPUT_DIM)
        y = mean + std * noise
        if OracleDiffusionModel is not None and isinstance(model, OracleDiffusionModel):
            if prediction_target == "x0":
                pred = batch["gt_xy"]
            elif prediction_target == "epsilon":
                pred = noise
            elif prediction_target == "score":
                pred = -noise / jnp.maximum(std, 1e-5)
            else:
                raise ValueError(
                    f"Unsupported prediction_target '{prediction_target}'"
                )
        else:
            pred = model(t, y, batch["context"])
        if prediction_target == "x0":
            target = batch["gt_xy"]
        elif prediction_target == "epsilon":
            target = noise
        elif prediction_target == "score":
            target = -noise / jnp.maximum(std, 1e-5)
        else:
            raise ValueError(f"Unsupported prediction_target '{prediction_target}'")
        err = (pred - target) ** 2
        loss = (err * batch["gt_xy_mask"]).sum() / jnp.maximum(
            batch["gt_xy_mask"].sum(), 1.0
        )
        return loss

    def on_fit_start(self) -> None:
        Path("checkpoints").mkdir(exist_ok=True)
        print(self.hparams)
        if self.load_last_checkpoint:
            try:
                self.model = eqx.tree_deserialise_leaves(
                    self.CHECKPOINT_PATH, self.model
                )
                print("Loaded weights")
            except:
                print("Didnt load weights")

    def on_fit_end(self):
        self._save_local_checkpoint()
        self._log_model_artifact()

    def on_train_epoch_end(self) -> None:
        if not self._should_run_metrics_this_epoch("train"):
            self._train_batches_for_metrics.clear()
            return
        if len(self.metrics_train) == 0 or len(self._train_batches_for_metrics) == 0:
            return
        enable_train_visualization = bool(
            self.vis.get("enable_train_visualization", False)
        )
        debug_denoiser_scale = bool(self.vis.get("debug_denoiser_scale", False))
        log_denoise_metric = bool(self.vis.get("log_denoise_metric", True))
        plot_vis_kwargs = self._plot_vis_kwargs()
        train_images = []
        self.metrics_train.reset()
        denoise_metric_vals = []
        for batch_idx, batch in enumerate(self._train_batches_for_metrics):
            if debug_denoiser_scale and batch_idx == 0:
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
                    first_pred_xy_plot = self._mask_pred_for_plot(
                        first_pred_xy, batch["gt_xy_mask"][0]
                    )
                    first_pred_xy_world = self._to_world_frame(
                        first_pred_xy_plot,
                        batch["origin_xy"][0],
                    )
                    img = plot_simulator_state(
                        viz_state,
                        batch_idx=0,
                        pred_xy=first_pred_xy_world,
                        **plot_vis_kwargs,
                    )
                    train_images.append(img)
        vals = self.metrics_train.compute()
        log_dict = {f"Train/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
        if len(denoise_metric_vals) > 0:
            log_dict["Train/OneStepDenoise_ADE"] = float(
                sum(denoise_metric_vals) / len(denoise_metric_vals)
            )
        self.log_dict(log_dict, prog_bar=True)
        if enable_train_visualization and len(train_images) > 0:
            self._log_images(
                f"Train scenarios and predictions/epoch_{self.current_epoch}",
                train_images,
            )
        print(f"[Train metrics] epoch={self.current_epoch}  " + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
        self._train_batches_for_metrics.clear()

    def _compute_one_step_denoise_ade(self, batch):
        sample_idx = 0
        t = jnp.array(min(float(self.t1) * 0.5, 1.0), dtype=jnp.float32)
        key = jr.PRNGKey(0)
        gt_xy = jnp.asarray(batch["gt_xy"][sample_idx])
        mean = gt_xy * jnp.exp(-0.5 * self.int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-self.int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, gt_xy.shape)
        y = mean + std * noise
        pred_raw = jnp.asarray(self.model(t, y, batch["context"][sample_idx]))
        pred = self._prediction_to_x0(pred_raw, y, t)
        gt_mask = jnp.asarray(batch["gt_xy_mask"][sample_idx])
        if pred.ndim + 1 == gt_xy.ndim and gt_xy.shape[0] == 1:
            gt_xy = jnp.squeeze(gt_xy, axis=0)
            gt_mask = jnp.squeeze(gt_mask, axis=0)
        ade_metric = self.metrics_train.metrics[0].__class__(name="tmp_ADE")
        ade_metric.update(pred, gt_xy, gt_mask)
        return float(jnp.asarray(ade_metric.compute()))

    def _debug_denoiser_scale(self, batch):
        sample_idx = 0
        t = jnp.array(min(float(self.t1) * 0.5, 1.0), dtype=jnp.float32)
        key = jr.PRNGKey(0)
        gt_xy = jnp.asarray(batch["gt_xy"][sample_idx])
        mean = gt_xy * jnp.exp(-0.5 * self.int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-self.int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, gt_xy.shape)
        y = mean + std * noise
        context = jnp.asarray(batch["context"][sample_idx])
        pred = jnp.asarray(self.model(t, y, context))
        target = jnp.asarray(self._prediction_to_target(gt_xy, noise, std))
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
            raise ValueError(
                f"Unexpected pred shape {pred.shape} for gt_xy {gt_xy.shape}"
            )
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

    def _debug_training_shapes(self, batch):
        sample_idx = 0
        gt_xy = jnp.asarray(batch["gt_xy"][sample_idx])
        context = jnp.asarray(batch["context"][sample_idx])
        gt_xy_mask = jnp.asarray(batch["gt_xy_mask"][sample_idx])

        t = jnp.array(min(float(self.t1) * 0.5, 1.0), dtype=jnp.float32)
        key = jr.PRNGKey(0)
        noise = jr.normal(key, gt_xy.shape)
        mean = gt_xy * jnp.exp(-0.5 * self.int_beta(t))
        std = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-self.int_beta(t)), 1e-5))
        y = mean + std * noise
        pred = jnp.asarray(self.model(t, y, context))

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

    def on_validation_epoch_end(self) -> None:
        self._save_local_checkpoint()
        if not self._should_run_metrics_this_epoch("val"):
            self._val_batches_for_metrics.clear()
            return
        if len(self.metrics_val) == 0 or len(self._val_batches_for_metrics) == 0:
            return

        images = []
        diffusion_video_frames = []
        enable_visualization = bool(self.vis.get("enable_visualization", False))
        plot_vis_kwargs = self._plot_vis_kwargs()
        self.metrics_val.reset()
        for batch_idx, batch in enumerate(self._val_batches_for_metrics):
            first_pred_xy, first_pred_path = self._update_metrics_for_batch(
                self.metrics_val,
                batch,
                return_first_prediction=batch_idx == 0,
            )

            if (
                enable_visualization
                and batch_idx == 0
                and "scenario" in batch
                and first_pred_xy is not None
            ):
                viz_state = batch["scenario"]
                if viz_state is not None:
                    first_pred_xy_plot = self._mask_pred_for_plot(
                        first_pred_xy, batch["gt_xy_mask"][0]
                    )
                    first_pred_xy_world = self._to_world_frame(
                        first_pred_xy_plot, batch["origin_xy"][0]
                    )
                    img = plot_simulator_state(
                        viz_state,
                        batch_idx=0,
                        pred_xy=first_pred_xy_world,
                        **plot_vis_kwargs,
                    )
                    images.append(img)
                    if len(diffusion_video_frames) == 0 and first_pred_path is not None:
                        first_pred_path_plot = self._mask_pred_for_plot(
                            first_pred_path, batch["gt_xy_mask"][0]
                        )
                        pred_path_np = np.asarray(
                            self._to_world_frame(
                                first_pred_path_plot,
                                batch["origin_xy"][0],
                            )
                        )
                        for s in range(pred_path_np.shape[0]):
                            frame = plot_simulator_state(
                                viz_state,
                                batch_idx=0,
                                pred_xy=pred_path_np[s],
                                **plot_vis_kwargs,
                            )
                            diffusion_video_frames.append(frame)
        if enable_visualization and "scenario" in batch and len(images) > 0:
            self._log_images(
                f"Scenarios and predictions/epoch_{self.current_epoch}",
                images,
            )
        if enable_visualization and len(diffusion_video_frames) > 0:
            video_np = np.stack(diffusion_video_frames, axis=0).astype(
                np.uint8
            )  # [T, H, W, C]
            gif_path = (
                Path(tempfile.gettempdir()) / f"diffusion_path_{self.current_epoch}.gif"
            )
            pil_frames = [Image.fromarray(frame) for frame in video_np]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(1000 / int(self.vis.get("sample_video_fps", 6)))),
                loop=0,
            )
            self._log_video(
                f"Diffusion trajectory video/epoch_{self.current_epoch}",
                gif_path,
            )

        vals = self.metrics_val.compute()
        log_dict = {
            f"{self.metrics_prefix}/{k}": float(jnp.asarray(v)) for k, v in vals.items()
        }
        self.log_dict(log_dict, prog_bar=True)
        self._val_batches_for_metrics.clear()

    def _save_local_checkpoint(self) -> None:
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(self.CHECKPOINT_PATH, self.model)

    def _log_model_artifact(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            return

        artifact_name = f"{getattr(logger, 'name', 'model')}-model"
        version = getattr(logger, "version", None)
        if version:
            artifact_name = f"{artifact_name}-{version}"

        self._upload_artifact(
            name=artifact_name,
            path=self.CHECKPOINT_PATH,
            metadata={
                "epoch": int(self.current_epoch),
                "global_step": int(self.global_step),
            },
        )

    def _update_metrics_for_batch(
        self, metrics, batch, return_first_prediction=False
    ):
        first_pred_xy = None
        first_pred_path = None
        num_solutions = self._num_metric_trajectory_samples()
        debug_metrics = bool(self.vis.get("debug_metrics", False))

        for sample_idx in range(batch["gt_xy"].shape[0]):
            if bool(self.vis.get("direct_prediction_eval", False)):
                pred_xy, pred_path = self._direct_predict(
                    batch["context"][sample_idx],
                    batch["gt_xy"][sample_idx].shape,
                    num_solutions=num_solutions,
                )
            else:
                pred_xy, pred_path = self.sample_multiple_sol(
                    batch["context"][sample_idx],
                    num_solutions=num_solutions,
                    predict_shape=batch["gt_xy"][sample_idx].shape,
                    oracle_gt_xy=(
                        batch["gt_xy"][sample_idx]
                        if self._oracle_enabled("use_for_sampling")
                        else None
                    ),
                )
            coord_scale = self._batch_coord_scale(batch, sample_idx)
            pred_xy_metric = self._to_metric_frame(
                pred_xy, coord_scale
            )
            pred_path_metric = self._to_metric_frame(
                pred_path, coord_scale
            )
            if debug_metrics:
                gt_xy = self._to_metric_frame(
                    batch["gt_xy"][sample_idx], coord_scale
                )
                gt_mask = batch["gt_xy_mask"][sample_idx][..., 0].astype(bool)
                pred_valid = pred_xy_metric[gt_mask]
                gt_valid = gt_xy[gt_mask]
                if pred_valid.size > 0:
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
            metrics.update(
                pred_xy_metric,
                self._to_metric_frame(
                    batch["gt_xy"][sample_idx], coord_scale
                ),
                batch["gt_xy_mask"][sample_idx],
            )
            if return_first_prediction and sample_idx == 0:
                first_pred_xy = pred_xy_metric
                first_pred_path = pred_path_metric

        return first_pred_xy, first_pred_path

    def sample_multiple_sol(
        self,
        context,
        num_solutions=1,
        predict_shape=None,
        save_full=False,
        oracle_gt_xy=None,
        y1_override=None,
    ):
        """
        Sample full reverse trajectory states for all agents.
        Returns:
            final_pred: [N, H, 2]
            all_preds: [S, N, H, 2] where S=num_solutions
        """
        if oracle_gt_xy is not None and self._oracle_sampling_mode() == "exact":
            pred_paths = jnp.repeat(oracle_gt_xy[None, ...], num_solutions, axis=0)
            return oracle_gt_xy, pred_paths

        self.sample_key, key = jr.split(self.sample_key)
        if y1_override is not None:
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
        else:
            y1_overrides = None

        sample_keys = jr.split(key, num_solutions)
        sample_one = lambda kk: self.sample_one_sol(
            self.model,
            self.int_beta,
            predict_shape,
            self.dt0,
            self.t1,
            context,
            save_full,
            oracle_gt_xy,
            y1_overrides,
            kk,
        )[0]
        if y1_overrides is None:
            pred_paths = jax.vmap(sample_one)(sample_keys)  # [S, *data_shape]
        else:
            pred_paths = jax.vmap(
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
        final_pred = jnp.mean(pred_paths, axis=0)       # mean over samples → [*data_shape]
        return final_pred, pred_paths

    @eqx.filter_jit
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
        """
        Return full reverse trajectory states sampled at `num_solutions` times.
        Output shape: [num_solutions, data_shape]
        """
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
