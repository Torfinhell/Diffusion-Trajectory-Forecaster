from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytorch_lightning as L
from hydra.utils import instantiate

from src.metrics import MetricCollection
from src.utils.eval import (
    image_log_name,
    log_images,
    mask_pred_for_plot,
    plot_vis_kwargs,
)
from src.visualization.viz import plot_simulator_state
from utils.data_utils import batch_transform_trajs_to_global_frame


class BaseTrainer(L.LightningModule):
    CHECKPOINT_ROOT = Path("checkpoints")

    def __init__(
        self,
        seed,
        load_best_checkpoint,
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
        self.load_best_checkpoint_flag = load_best_checkpoint
        self.samples = 10
        self.global_step_ = 0
        self.metrics_train = MetricCollection(
            [instantiate(m) for m in self.metrics.train]
        )
        self.metrics_val = MetricCollection([instantiate(m) for m in self.metrics.val])
        self.metrics_test = MetricCollection([instantiate(m) for m in self.metrics.val])
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

    def clip_optimizer(self, optimizer):
        transforms = []
        if self.grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip))
        transforms.append(optimizer)
        return optax.chain(*transforms)

    def _update_metrics_for_batch(self, metrics, batch):
        gt_xy_batch = batch["agent_future"][:, ..., :2]
        future_valid_batch = batch["agent_future_valid"]
        agents_coeffs_batch = batch["agents_coeffs"]
        batch_size = gt_xy_batch.shape[0]
        pred_xy_batch = self.sample_batch_sol(
            self.model,
            self.diffusion_sampler,
            gt_xy_batch.shape[1:],
            batch,
            batch_size=batch_size,
        )
        for sample_idx in range(batch_size):
            gt_xy = gt_xy_batch[sample_idx]
            future_valid = future_valid_batch[sample_idx]
            pred_xy = pred_xy_batch[sample_idx]
            metrics.update(
                pred_xy,
                gt_xy,
                agents_coeffs_batch[sample_idx],
                future_valid,
            )
        return pred_xy_batch

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
                jnp.asarray(timestep, dtype=x_t.dtype),
                x_t,
                batch,
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

    def _log_validation_visualizations(self, batch, sampled_trajs):
        enable_visualization = bool(self.vis.get("enable_visualization", False))
        has_scenarios = "scenario" in batch and batch["scenario"] is not None
        if not enable_visualization or not has_scenarios:
            return

        images = []
        plot_kwargs = plot_vis_kwargs(self)
        num_samples = min(int(self.vis.get("num_samples", 0)), sampled_trajs.shape[0])
        for i in range(num_samples):
            scenario = batch["scenario"][i]
            if scenario is None:
                continue
            pred_xy_plot = mask_pred_for_plot(
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
            log_images(
                self,
                image_log_name("val", "predictions"),
                images,
            )

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        metric_every = max(
            1, int(self.trainer_cfg.get("train_metric_every_n_epochs", 1))
        )
        should_run_metrics = (
            batch_idx == 0
            and len(self.metrics_train) > 0
            and (
                (self.current_epoch + 1) % metric_every == 0 or self.current_epoch == 0
            )
        )
        if should_run_metrics:
            self.metrics_train.reset()
            sampled_trajs = self._update_metrics_for_batch(self.metrics_train, batch)
            vals = self.metrics_train.compute()
            log_dict = {f"train/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch["agent_future"].shape[0],
            )
            self._log_validation_visualizations(batch, sampled_trajs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        metric_every = max(1, int(self.trainer_cfg.get("val_metric_every_n_epochs", 1)))
        should_run_metrics = (
            batch_idx == 0
            and len(self.metrics_val) > 0
            and (
                (self.current_epoch + 1) % metric_every == 0 or self.current_epoch == 0
            )
        )
        if should_run_metrics:
            self.metrics_val.reset()
            sampled_trajs = self._update_metrics_for_batch(self.metrics_val, batch)
            vals = self.metrics_val.compute()
            log_dict = {f"val/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch["agent_future"].shape[0],
            )
            self._log_validation_visualizations(batch, sampled_trajs)
        return loss

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

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

        log_metrics = {f"{kind}/loss_step": float(jnp.asarray(step_out["loss"]))}
        if is_train:
            log_metrics[f"train/grad_norm"] = float(jnp.asarray(step_out["grad_norm"]))
            log_metrics[f"train/update_norm"] = float(
                jnp.asarray(step_out["update_norm"])
            )
            log_metrics[f"train/param_norm"] = float(
                jnp.asarray(step_out["param_norm"])
            )

        self.log_dict(
            log_metrics,
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
            grad_fn = eqx.filter_value_and_grad(BaseTrainer.batch_loss_fn, has_aux=True)
            loss, grads = grad_fn(model, diffusion_sampler, loss_fn, batch, key)
            grad_norm = optax.global_norm(grads)
            updates, opt_state = opt_update(grads, opt_state)
            update_norm = optax.global_norm(updates)
            model = eqx.apply_updates(model, updates)
            param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        else:
            loss = BaseTrainer.batch_loss_fn(
                model, diffusion_sampler, loss_fn, batch, key
            )
            grad_norm = None
            update_norm = None
            param_norm = None

        # key = jr.split(key, 1)[0]
        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "model": model,
            # "key": key,
            "opt_state": opt_state,
        }

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
                **single_sample_dict,
            )

        losses = jax.vmap(mapped_fn)(batch, loss_keys)
        return jnp.mean(losses)
