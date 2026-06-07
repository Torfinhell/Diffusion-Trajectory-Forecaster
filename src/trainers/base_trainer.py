from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytorch_lightning as L
from hydra.utils import instantiate

from src.data_module.agent_path import AgentPath
from src.losses import KDLoss, MseActionFullLoss, MseActionLoss, MseXYLoss
from src.utils import (
    load_best_checkpoint,
    log_model_artifact,
    maybe_save_best_checkpoint,
)
from src.visualization.viz import plot_simulator_state


class BaseTrainer(L.LightningModule):
    CHECKPOINT_ROOT = Path("checkpoints")

    def __init__(
        self,
        model,
        loss_fn,
        optim,
        opt_state,
        diffusion_sampler,
        cfg_metrics,
        vis_cfg,
        key,
        train_key,
        loader_key,
        sample_key,
        extract_actions,
        action_len,
        log_metrics_every_batch=10,
        train_metric_every_n_epochs=1,
        max_train_metric_batches=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss_fn",
                "optim",
                "opt_state",
                "key",
                "train_key",
                "loader_key",
                "sample_key",
            ]
        )
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.opt_state = opt_state
        self.diffusion_sampler = diffusion_sampler
        self.key = key
        self.train_key = train_key
        self.loader_key = loader_key
        self.sample_key = sample_key
        self.extract_actions = bool(extract_actions)
        self.action_len = int(action_len)
        self.log_metrics_every_batch = log_metrics_every_batch
        self.train_metric_every_n_epochs = int(train_metric_every_n_epochs)
        self.max_train_metric_batches = (
            int(max_train_metric_batches) if max_train_metric_batches is not None else None
        )
        if self.extract_actions:
            assert isinstance(loss_fn, (MseActionLoss, MseActionFullLoss, KDLoss))
        else:
            assert isinstance(loss_fn, MseXYLoss)
        self._init_trainer_state(cfg_metrics, vis_cfg, kwargs)

    def _init_trainer_state(self, cfg_metrics, vis_cfg, trainer_cfg):
        self.automatic_optimization = False
        self.metrics = cfg_metrics
        self.vis = vis_cfg
        self.trainer_cfg = trainer_cfg
        self.debug = bool(trainer_cfg.pop("debug", False))
        grad_clip = trainer_cfg.pop("grad_clip", None)
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
            float("inf") if self.best_checkpoint_mode == "min" else float("-inf")
        )
        self.best_checkpoint_epoch = -1
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

    def on_validation_epoch_end(self):
        if not self.debug or self.trainer.sanity_checking:
            return
        metrics = {}
        for attr in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
            metrics.update(getattr(self.trainer, attr, None) or {})
        maybe_save_best_checkpoint(self, metrics)

    def on_fit_end(self):
        if not self.debug:
            return
        if bool(self.trainer_cfg.get("load_best_checkpoint", False)):
            load_best_checkpoint(self)
        log_model_artifact(self)

    def configure_optimizers(self):
        return []

    @staticmethod
    def _past_ref_coords(agent_past, agent_past_valid):
        # (A, T, 5), (A, T) -> (A, 5): last valid past state per agent
        t = agent_past.shape[-2]
        idxs = jnp.arange(t)
        last_valid = jnp.max(jnp.where(agent_past_valid, idxs[None, :], -1), axis=-1)
        last_valid = jnp.maximum(last_valid, 0)
        return agent_past[jnp.arange(agent_past.shape[0]), last_valid]

    def _loss_scales(self):
        return (
            getattr(self.loss_fn, "accel_scale", 1.0),
            getattr(self.loss_fn, "yaw_rate_scale", 0.15),
            getattr(self.loss_fn, "coord_scale", 1.0),
        )

    def _decode_sample(self, sampled, past_path: AgentPath, future_path: AgentPath):
        accel_scale, yaw_rate_scale, coord_scale = self._loss_scales()
        if self.extract_actions:
            return past_path.decode_action_sample(
                sampled, accel_scale=accel_scale, yaw_rate_scale=yaw_rate_scale
            )
        return future_path.decode_xy_sample(
            sampled, coord_scale=coord_scale, past_path=past_path
        )

    def training_step(self, batch, batch_idx):
        self._step(batch, "train", batch_idx)
        return None

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val", batch_idx)
        return None

    def test_step(self, batch, batch_idx):
        self._step(batch, "test", batch_idx)
        return None

    def _step(self, batch, kind, batch_idx=None):
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
            debug=self.debug,
            action_len=self.action_len,
            opt_state=self.opt_state if is_train else None,
            opt_update=self.optim.update if is_train else None,
        )

        if is_train:
            self.model = step_out["model"]
            self.opt_state = step_out["opt_state"]
            self._apply_distill_proj_updates(step_out)
        self.log_dict(
            {
                f"{kind}/{key}": float(jnp.asarray(value))
                for key, value in step_out.items()
                if key
                not in {"model", "opt_state", "projectors", "past_path", "future_path"}
                and value is not None
            },
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=batch["agent_future"].shape[0],
        )
        if is_train:
            self.global_step_ += 1

        metrics_cfg = {
            "train": self.metrics_train,
            "val": self.metrics_val,
            "test": self.metrics_test,
        }.get(kind)
        epoch_gate = self.current_epoch % self.train_metric_every_n_epochs == 0
        max_tmb = self.max_train_metric_batches
        batch_limit_gate = (
            kind != "train"
            or max_tmb is None
            or batch_idx is None
            or batch_idx < max_tmb
        )
        if (
            batch_idx is None
            or metrics_cfg is None
            or not epoch_gate
            or not batch_limit_gate
            or batch_idx % self.log_metrics_every_batch != 0
        ):
            return step_out["loss"]

        batch_size = batch["agent_future"].shape[0]
        data_shape = self.model.out_shape
        self.sample_key, key = jr.split(self.sample_key)
        sample_keys = jr.split(key, batch_size)
        sampled_pred_batch = self.sample_batch_sol(
            self.model,
            self.diffusion_sampler,
            data_shape,
            batch,
            sample_keys,
            self.action_len,
        )

        def process_one(sample, agent_past, agent_future, agent_past_valid):
            ref_coords = BaseTrainer._past_ref_coords(agent_past, agent_past_valid)
            past = AgentPath(agent_past, self.action_len, ref_coords=ref_coords)
            future_local = AgentPath(
                agent_future, self.action_len, ref_coords=ref_coords
            )
            pred_xy = self._decode_sample(sample, past, future_local)
            gt_xy = future_local.to_local()[..., :2]
            return pred_xy, gt_xy

        pred_xy_batch, gt_xy_batch = jax.vmap(process_one)(
            sampled_pred_batch, batch["agent_past"], batch["agent_future"],
            batch["agent_past_valid"],
        )
        batch.update(
            {
                "pred_xy": pred_xy_batch,
                "gt_xy": gt_xy_batch,
                "future_valid": batch["agent_future_valid"],
            }
        )
        vals = metrics_cfg(**batch)
        self.log_dict(
            {f"{kind}/{k}": float(jnp.asarray(v)) for k, v in vals.items()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if kind == "val" and bool(self.trainer_cfg.get("log_validation", True)):
            self._log_validation_visualizations("val", batch, pred_xy_batch)
        return step_out["loss"]

    def _apply_distill_proj_updates(self, step_out):
        pass

    @staticmethod
    def make_step(
        model,
        diffusion_sampler,
        loss_fn,
        batch,
        key,
        train,
        action_len,
        debug=False,
        opt_state=None,
        opt_update=None,
    ):
        if train:

            def loss_with_aux(model, diffusion_sampler, loss_fn, batch, key, debug):
                mean_dict = BaseTrainer.batch_loss_fn(
                    model,
                    diffusion_sampler,
                    loss_fn,
                    batch,
                    key,
                    action_len,
                    debug,
                )
                return mean_dict["loss"], mean_dict

            grad_fn = eqx.filter_value_and_grad(loss_with_aux, has_aux=True)
            (_, mean_dict), grads = grad_fn(
                model, diffusion_sampler, loss_fn, batch, key, debug
            )
            grad_norm = optax.global_norm(grads)

            # Extract the inexact arrays (parameters) to pass to the optimizer update function
            params = eqx.filter(model, eqx.is_inexact_array)
            updates, opt_state = opt_update(grads, opt_state, params=params)

            update_norm = optax.global_norm(updates)
            model = eqx.apply_updates(model, updates)
            param_norm = optax.global_norm(params)
        else:
            mean_dict = BaseTrainer.batch_loss_fn(
                model, diffusion_sampler, loss_fn, batch, key, action_len, debug
            )
            grad_norm = None
            update_norm = None
            param_norm = None

        step_out = {
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            **mean_dict,
        }
        if train:
            step_out["model"] = model
            step_out["opt_state"] = opt_state
        return step_out

    @staticmethod
    def batch_loss_fn(
        model,
        diffusion_sampler,
        loss_fn,
        batch,
        key,
        action_len,
        debug=False,
    ):
        batch = {name: value for name, value in batch.items() if name != "scenario"}
        loss_keys = jr.split(key, jax.tree_util.tree_leaves(batch)[0].shape[0])

        def mapped_fn(single_sample_dict, single_key):
            ref_coords = BaseTrainer._past_ref_coords(
                single_sample_dict["agent_past"], single_sample_dict["agent_past_valid"]
            )
            past = AgentPath(single_sample_dict["agent_past"], action_len, ref_coords=ref_coords)
            future = AgentPath(single_sample_dict["agent_future"], action_len, ref_coords=ref_coords)
            full_loss = loss_fn(
                model=model,
                diffusion_sampler=diffusion_sampler,
                past_path=past,
                future_path=future,
                key=single_key,
                debug=debug,
                **single_sample_dict,
            )
            return full_loss

        loss_dicts = jax.vmap(mapped_fn)(batch, loss_keys)
        return jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_dicts)

    @staticmethod
    @eqx.filter_jit
    def sample_one_sol(
        model,
        diffusion_sampler,
        data_shape,
        batch,
        key,
        action_len: int,
        save_full=False,
    ):
        step_keys = jr.split(key, diffusion_sampler.num_steps + 1)
        x = jr.normal(step_keys[0], data_shape)
        timesteps = jnp.arange(diffusion_sampler.num_steps - 1, -1, -1, dtype=jnp.int32)

        def scan_step(x_t, inputs):
            timestep, step_key = inputs
            timestep_arr = jnp.asarray(timestep, dtype=jnp.int32)
            past = AgentPath(batch["agent_past"], action_len, ref_coords=BaseTrainer._past_ref_coords(batch["agent_past"], batch["agent_past_valid"]))
            model_output = model(timestep_arr, x_t, past_path=past, **batch)
            x_prev = diffusion_sampler.step(step_key, model_output, timestep_arr, x_t)
            return x_prev, x_prev

        x, path = jax.lax.scan(scan_step, x, (timesteps, step_keys[1:]))
        return path if save_full else x

    @staticmethod
    @eqx.filter_jit
    def sample_batch_sol(
        model,
        diffusion_sampler,
        data_shape,
        batch,
        sample_keys,
        action_len: int,
        save_full=False,
    ):
        safe_batch = {k: v for k, v in batch.items() if k != "scenario"}
        sample_fn = lambda sample_key, single_batch: BaseTrainer.sample_one_sol(
            model,
            diffusion_sampler,
            data_shape,
            single_batch,
            sample_key,
            action_len,
            save_full=save_full,
        )
        return jax.vmap(sample_fn)(sample_keys, safe_batch)

    def _log_validation_visualizations(self, split, batch, pred_xy_batch):
        enable_visualization = bool(self.vis.get("enable_visualization", False))
        has_scenarios = "scenario" in batch and batch["scenario"] is not None
        if not enable_visualization or not has_scenarios:
            return

        images = []
        num_samples = min(int(self.vis.get("num_samples", 0)), pred_xy_batch.shape[0])
        for i in range(num_samples):
            scenario = batch["scenario"][i]
            if scenario is None:
                continue
            pred_xy_plot = self._mask_pred_for_plot(
                pred_xy_batch[i], batch["agents_coeffs"][i]
            )
            past_path = AgentPath(batch["agent_past"][i], self.action_len, ref_coords=BaseTrainer._past_ref_coords(batch["agent_past"][i], batch["agent_past_valid"][i]))

            anchor = past_path.ref_coords
            x0 = anchor[..., 0]
            y0 = anchor[..., 1]
            theta0 = anchor[..., 2]
            cos_t = jnp.cos(theta0)
            sin_t = jnp.sin(theta0)
            local = jnp.asarray(pred_xy_plot)
            g_x = (
                local[..., 0] * cos_t[..., None]
                - local[..., 1] * sin_t[..., None]
                + x0[..., None]
            )
            g_y = (
                local[..., 0] * sin_t[..., None]
                + local[..., 1] * cos_t[..., None]
                + y0[..., None]
            )
            pred_xy_world = jnp.stack([g_x, g_y], axis=-1)
            images.append(
                plot_simulator_state(
                    scenario,
                    pred_xy=pred_xy_world,
                    **self.vis,
                )
            )
        if images:
            self._log_images(self._image_log_name(split, "predictions"), images)

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
