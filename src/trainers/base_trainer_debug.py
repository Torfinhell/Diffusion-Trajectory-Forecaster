from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytorch_lightning as L
from hydra.utils import instantiate

from src.metrics import MetricFnCollection
from src.utils import (
    load_best_checkpoint,
    log_model_artifact,
    maybe_save_best_checkpoint,
)
from src.utils.data_utils import (
    batch_transform_trajs_to_global_frame,
    predictions_to_local_xy,
)
from src.visualization.viz import plot_simulator_state


class BaseTrainerDebug(L.LightningModule):
    CHECKPOINT_ROOT = Path("checkpoints")

    def __init__(
        self,
        seed,
        cfg_metrics,
        vis_cfg,
        model,
        loss,
        optimizer,
        scheduler=None,
        diffusion_sampler=None,
        grad_clip=None,
        trainer_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        del kwargs
        self.automatic_optimization = False
        self.key = jax.random.PRNGKey(seed)
        self.key, key_model, self.train_key, self.loader_key, self.sample_key = (
            jax.random.split(self.key, 5)
        )
        self.metrics = cfg_metrics
        self.vis = vis_cfg
        self.trainer_cfg = trainer_cfg or {}
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        if self.grad_clip is not None and self.grad_clip <= 0:
            self.grad_clip = None
        self.best_checkpoint_metric = str(
            self.trainer_cfg.get("best_checkpoint_metric", "val/loss")
        )
        self.best_checkpoint_mode = str(
            self.trainer_cfg.get("best_checkpoint_mode", "min")
        ).lower()
        if self.best_checkpoint_mode not in {"min", "max"}:
            raise ValueError(
                "trainer.best_checkpoint_mode must be either 'min' or 'max'."
            )
        self.best_checkpoint_score = (
            float("inf")
            if self.best_checkpoint_mode == "min"
            else float("-inf")
        )
        self.best_checkpoint_epoch = -1
        self.samples = 10
        self.global_step_ = 0
        self.metrics_train = (
            instantiate(self.metrics.train)
            if getattr(self.metrics, "train", None)
            else None
        )
        self.metrics_val = (
            instantiate(self.metrics.val)
            if getattr(self.metrics, "val", None)
            else None
        )
        self.metrics_test = self.metrics_val
        self.model = instantiate(model, key=key_model)
        self.loss_fn = instantiate(loss)
        self.learning_rate_schedule = (
            instantiate(scheduler) if scheduler is not None else None
        )
        self.diffusion_sampler = (
            instantiate(diffusion_sampler) if diffusion_sampler is not None else None
        )
        optimizer_args = {}
        if self.learning_rate_schedule is not None:
            optimizer_args["learning_rate"] = self.learning_rate_schedule
        optimizer_transform = instantiate(optimizer, **optimizer_args)
        self.optim = self.clip_optimizer(optimizer_transform)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

    def configure_optimizers(self):
        return []

    def clip_optimizer(self, optimizer):
        transforms = []
        if self.grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip))
        transforms.append(optimizer)
        return optax.chain(*transforms)

    def _update_metrics_for_batch(self, metrics: MetricFnCollection, batch):
        gt_xy_batch = batch["agent_future"][:, ..., :2]
        batch_size = gt_xy_batch.shape[0]
        sampled_pred_batch = self.sample_batch_sol(
            self.model,
            self.diffusion_sampler,
            batch["actions_future"].shape[1:],
            batch,
            batch_size=batch_size,
        )
        pred_xy_batch, _ = predictions_to_local_xy(
            sampled_pred_batch,
            agent_past=batch["agent_past"],
            origin_vel=batch["origin_vel"],
            agent_future=batch["agent_future"],
            actions_future=batch["actions_future"],
            accel_scale=self.loss_fn.accel_scale,
            yaw_rate_scale=self.loss_fn.yaw_rate_scale,
        )
        batch["pred_xy"] = pred_xy_batch
        batch["gt_xy"] = gt_xy_batch
        batch["future_valid"] = batch["agent_future_valid"]
        vals = metrics(**batch)
        return pred_xy_batch, vals

    def sample_one_sol(
        self,
        model,
        diffusion_sampler,
        data_shape,
        batch,
        save_full=False,
        key=None,
    ):
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        step_keys = jr.split(key, diffusion_sampler.num_steps + 1)
        x = jr.normal(step_keys[0], data_shape)
        timesteps = jnp.arange(diffusion_sampler.num_steps - 1, -1, -1, dtype=jnp.int32)

        def scan_step(x_t, inputs):
            timestep, step_key = inputs
            timestep_arr = jnp.asarray(timestep, dtype=jnp.int32)
            model_output = model(
                timestep_arr,
                x_t,
                **batch,
            )
            x_prev = diffusion_sampler.step(
                step_key,
                model_output,
                timestep_arr,
                x_t,
            )
            return x_prev, x_prev

        x, path = jax.lax.scan(scan_step, x, (timesteps, step_keys[1:]))
        if save_full:
            return path
        return x

    def sample_batch_sol(
        self,
        model,
        diffusion_sampler,
        data_shape,
        batch,
        batch_size,
        save_full=False,
    ):
        self.sample_key, key = jr.split(self.sample_key)
        sample_keys = jr.split(key, batch_size)
        sample_fn = lambda sample_key, batch: self.sample_one_sol(
            model,
            diffusion_sampler,
            data_shape,
            batch,
            save_full=save_full,
            key=sample_key,
        )
        safe_vmap_batch = {k: v for k, v in batch.items() if k != "scenario"}
        return jax.vmap(sample_fn)(sample_keys, safe_vmap_batch)

    def _log_validation_visualizations(self, split, batch, sampled_trajs):
        enable_visualization = bool(self.vis.get("enable_visualization", False))
        has_scenarios = "scenario" in batch and batch["scenario"] is not None
        if not enable_visualization or not has_scenarios:
            return

        images = []
        plot_kwargs = self._plot_vis_kwargs()
        num_samples = min(int(self.vis.get("num_samples", 0)), sampled_trajs.shape[0])
        for i in range(num_samples):
            scenario = batch["scenario"][i]
            if scenario is None:
                continue
            pred_xy_plot = self._mask_pred_for_plot(
                sampled_trajs[i], batch["agents_coeffs"][i]
            )
            pred_xy_world = batch_transform_trajs_to_global_frame(
                pred_xy_plot,
                origin_xy=batch["origin_xy"][i],
                origin_theta=batch["origin_theta"][i],
            )

            images.append(
                plot_simulator_state(
                    scenario,
                    pred_xy=pred_xy_world,
                    **plot_kwargs,
                )
            )

        if images:
            self._log_images(self._image_log_name(split, "predictions"), images)

    def _plot_vis_kwargs(self):
        excluded = {
            "debug_metrics",
            "debug_denoiser_scale",
            "direct_prediction_eval",
            "num_samples",
            "num_trajectory_samples",
            "sample_steps",
            "sample_video_fps",
            "enable_visualization",
            "enable_train_visualization",
        }
        return {k: v for k, v in self.vis.items() if k not in excluded}

    @staticmethod
    def _image_log_name(split, name):
        return f"images/{split}_{name}"

    @staticmethod
    def _mask_pred_for_plot(pred_xy, agents_coeffs):
        pred_xy = jnp.asarray(pred_xy)
        agents_coeffs = jnp.asarray(agents_coeffs)
        while agents_coeffs.ndim > pred_xy.ndim - 2:
            agents_coeffs = jnp.squeeze(agents_coeffs, axis=0)
        while agents_coeffs.ndim < pred_xy.ndim - 2:
            agents_coeffs = agents_coeffs[None, ...]
        valid_agents = (agents_coeffs > 0)[..., None, None]
        return jnp.where(valid_agents, pred_xy, jnp.nan)

    def _log_images(self, key, images):
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "log_image"):
            return
        logger.log_image(key=key, images=images, step=int(self.global_step_))

    def training_step(self, batch, batch_idx):
        self._step(batch, "train")
        metric_every = max(
            1, int(self.trainer_cfg.get("train_metric_every_n_epochs", 1))
        )
        should_run_metrics = len(self.metrics_train) > 0 and (
            (self.current_epoch + 1)% metric_every == 0
        )
        if should_run_metrics:
            sampled_trajs, vals = self._update_metrics_for_batch(
                self.metrics_train, batch
            )
            log_dict = {f"train/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch["agent_future"].shape[0],
            )
            if bool(self.trainer_cfg.get("log_validation", True)) and batch_idx == 0:
                self._log_validation_visualizations("train", batch, sampled_trajs)
        return None

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")
        metric_every = max(1, int(self.trainer_cfg.get("val_metric_every_n_epochs", 1)))
        should_run_metrics = len(self.metrics_val) > 0 and (
            (self.current_epoch + 1) % metric_every == 0
        )
        if should_run_metrics:
            sampled_trajs, vals = self._update_metrics_for_batch(
                self.metrics_val, batch
            )
            log_dict = {f"val/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch["agent_future"].shape[0],
            )
            if bool(self.trainer_cfg.get("log_validation", True)) and batch_idx == 0:
                self._log_validation_visualizations("val", batch, sampled_trajs)
        return None

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = {}
        for attr in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
            values = getattr(self.trainer, attr, None)
            metrics.update(values)
        maybe_save_best_checkpoint(self, metrics)

    def on_fit_end(self):
        if bool(self.trainer_cfg.get("load_best_checkpoint", False)):
            load_best_checkpoint(self)
        log_model_artifact(self)

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")
        sampled_trajs, vals = self._update_metrics_for_batch(
                self.metrics_test, batch
            )
        log_dict = {f"test/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["agent_future"].shape[0],
        )
        if bool(self.trainer_cfg.get("log_validation", True)) and batch_idx == 0:
            self._log_validation_visualizations("test", batch, sampled_trajs)

    def _step(self, batch, kind):
        is_train = kind == "train"
        if is_train:
            step_key, self.train_key = jr.split(self.train_key)
        else:
            step_key, self.loader_key = jr.split(self.loader_key)

        step_out = self.make_step(
            model=self.model,
            diffusion_sampler=self.diffusion_sampler,
            loss_fn=self.loss_fn,
            batch=batch,
            key=step_key,
            train=is_train,
            opt_state=self.opt_state if is_train else None,
            opt_update=self.optim.update if is_train else None,
        )

        if is_train:
            self.model = step_out["model"]
            self.opt_state = step_out["opt_state"]

        log_output = {
            f"{kind}/{key}": float(jnp.asarray(value))
            for key, value in step_out.items()
            if key not in {"model", "opt_state"} and value is not None
        }

        self.log_dict(
            log_output,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=batch["agent_future"].shape[0],
        )

        if is_train:
            self.global_step_ += 1
        return step_out["loss"]

    @staticmethod
    @eqx.filter_jit
    def make_step(
        model,
        diffusion_sampler,
        loss_fn,
        batch,
        key,
        train,
        opt_state=None,
        opt_update=None,
    ):
        # TODO how to sample steps?
        if train:
            grad_fn = eqx.filter_value_and_grad(
                BaseTrainerDebug.batch_loss_fn, has_aux=True
            )
            (loss, stats), grads = grad_fn(
                model, diffusion_sampler, loss_fn, batch, key
            )
            grad_norm = optax.global_norm(grads)
            updates, opt_state = opt_update(grads, opt_state)
            update_norm = optax.global_norm(updates)
            model = eqx.apply_updates(model, updates)
            param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        else:
            loss, stats = BaseTrainerDebug.batch_loss_fn(
                model, diffusion_sampler, loss_fn, batch, key
            )
            grad_norm = None
            update_norm = None
            param_norm = None

        step_out = {
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "loss": loss,
            **stats,
        }
        if train:
            step_out["model"] = model
            step_out["opt_state"] = opt_state
        return step_out

    @eqx.filter_jit
    def batch_loss_fn(model, diffusion_sampler, loss_fn, batch, key):
        batch = {k: v for k, v in batch.items() if k != "scenario"}
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        loss_keys = jr.split(key, batch_size)

        def mapped_fn(single_sample_dict, single_key):
            return loss_fn(
                model=model,
                diffusion_sampler=diffusion_sampler,
                key=single_key,
                debug=True,
                **single_sample_dict,
            )

        losses, stats = jax.vmap(mapped_fn)(batch, loss_keys)
        mean_stats = jax.tree.map(lambda x: jnp.mean(x, axis=0), stats)
        return jnp.mean(losses, axis=0), mean_stats
