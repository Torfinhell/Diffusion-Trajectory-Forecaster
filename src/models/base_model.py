from functools import partial
from pathlib import Path

from torchgen import model

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytorch_lightning as L
from hydra.utils import instantiate

from src.metrics import MetricCollection, MetricTracker


class Basetreainer(L.LightningModule):
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
        grad_clip=None,
        trainer_cfg=None,
        prediction_target="x0",
        t0=1e-3,
        t1=2.0,
        dt0=0.01,
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
        self.train_loss_tracker = MetricTracker("train_loss")
        self.val_loss_tracker = MetricTracker("val_loss", "val_proxy_loss")
        self.val_metric_batches = int(self.trainer_cfg.get("val_metric_batches", 3))
        self._val_batches_for_metrics = []
        self.train_metric_batches = int(self.trainer_cfg.get("train_metric_batches", 3))
        self._train_batches_for_metrics = []

        self.prediction_target = str(prediction_target).lower()
        if self.prediction_target not in {"x0", "epsilon", "score"}:
            raise ValueError(
                f"Unsupported prediction_target '{prediction_target}'. "
                "Use one of: x0, epsilon, score."
            )
        self.int_beta = lambda t: t
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.dt0 = float(dt0)
        self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))

        self.model = instantiate(model, key=key_model)
        self.loss_fn = instantiate(loss)
        self.learning_rate_schedule = (
            instantiate(scheduler) if scheduler is not None else None
        )
        optimizer_args = {}
        if self.learning_rate_schedule is not None:
            optimizer_args["learning_rate"] = self.learning_rate_schedule
        optimizer_transform = instantiate(optimizer, **optimizer_args)
        self.optim = self.clip_optimizer(optimizer_transform)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))
        self.data_shape = None

    def clip_optimizer(self, optimizer):
        transforms = []
        if self.grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip))
        transforms.append(optimizer)
        return optax.chain(*transforms)

    def configure_optimizers(self):
        return []

    def training_step(self, batch):
        step_out = Basetreainer.make_train_step(
            model=self.model,
            loss_fn=self.loss_fn,
            int_beta=self.int_beta,
            batch=batch,
            timesteps=self.t1,
            key=self.train_key,
            opt_state=self.opt_state,
            opt_update=self.optim.update,
        )
        self.model = step_out["model"]
        self.train_key = step_out["key"]
        self.opt_state = step_out["opt_state"]
        self.train_loss_tracker.update("train_loss", jnp.asarray(step_out["loss"]))
        log_metrics = self._log_step_metrics(
            "train",
            step_out["loss"],
            step_out["train_stats"],
            grad_norm=step_out["grad_norm"],
            update_norm=step_out["update_norm"],
            param_norm=step_out["param_norm"],
        )
        self.log_dict(
            log_metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
        self.global_step_ += 1
        return None

    def validation_step(self, batch):
        self.loader_key, val_key = jr.split(self.loader_key)
        step_out = self.make_eval_step(
            model=self.model,
            loss_fn=self.loss_fn,
            int_beta=self.int_beta,
            batch=batch,
            timesteps=self.t1,
            key=val_key,
        )
        self.loader_key = step_out["key"]
        self.val_loss_tracker.update("val_loss", jnp.asarray(step_out["loss"]))
        log_metrics = self._log_step_metrics(
            "val",
            step_out["loss"],
            step_out["train_stats"],
        )
        self.log_dict(
            log_metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

    def test_step(self, batch):
        self.loader_key, test_key = jr.split(self.loader_key)
        step_out = self.make_eval_step(
            model=self.model,
            loss_fn=self.loss_fn,
            int_beta=self.int_beta,
            batch=batch,
            timesteps=self.t1,
            key=test_key,
        )
        self.loader_key = step_out["key"]
        log_metrics = self._log_step_metrics(
            "test",
            step_out["loss"],
            step_out["train_stats"],
        )
        self.log_dict(
            log_metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

    @classmethod
    def _log_step_metrics(
        cls,
        kind,
        loss,
        train_stats,
        grad_norm=None,
        update_norm=None,
        param_norm=None,
    ):
        log_metrics = {f"{kind}/loss_step": float(jnp.asarray(loss))}
        if grad_norm is not None:
            log_metrics[f"{kind}/grad_norm"] = float(jnp.asarray(grad_norm))
        if update_norm is not None:
            log_metrics[f"{kind}/update_norm"] =float(jnp.asarray(update_norm))
        if param_norm is not None:
            log_metrics[f"{kind}/param_norm"] = float(jnp.asarray(param_norm))
        if train_stats is not None:
            for stat_key, stat_value in train_stats.items():
                log_metrics[f"{kind}/{stat_key}"] = float(jnp.asarray(stat_value))
        return log_metrics

    @staticmethod
    @eqx.filter_jit
    def make_train_step(
        model,
        loss_fn,
        int_beta,
        batch,
        timesteps,
        key,
        opt_state=None,
        opt_update=None,
    ):
        grad_fn = eqx.filter_value_and_grad(
            Basetreainer.batch_loss_fn, has_aux=True
        )
        (loss, train_stats), grads = grad_fn(
            model, loss_fn, int_beta, batch, timesteps, key
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = opt_update(grads, opt_state)
        update_norm = optax.global_norm(updates)
        model = eqx.apply_updates(model, updates)
        param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))

        key = jr.split(key, 1)[0]
        return {
            "loss": loss,
            "train_stats": train_stats,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "model": model,
            "key": key,
            "opt_state": opt_state,
        }
    
    @staticmethod
    @eqx.filter_jit
    def make_eval_step(
        model,
        loss_fn,
        int_beta,
        batch,
        timesteps,
        key,
        opt_state=None,
        opt_update=None,
    ):
        loss, train_stats = Basetreainer.batch_loss_fn(
            model, loss_fn, int_beta, batch, timesteps, key
        )
        grad_norm = None
        update_norm = None
        param_norm = None
        key = jr.split(key, 1)[0]
        return {
            "loss": loss,
            "train_stats": train_stats,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "model": model,
            "key": key,
            "opt_state": opt_state,
        }

    @staticmethod
    def batch_loss_fn(model, loss_fn, int_beta, batch, t1, key):
        batch_size = batch["agent_future"].shape[0]
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        t_min = min(0.1, float(t1) * 0.5)
        t = jr.uniform(tkey, (batch_size,), minval=t_min, maxval=t1)
        sample_loss_fn = partial(loss_fn, model, int_beta)
        sample_loss_fn = jax.vmap(sample_loss_fn)
        losses, stats = sample_loss_fn(batch, t, losskey)
        mean_stats = jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0), stats
        )
        return jnp.mean(losses), mean_stats

    # def sample_multiple_sol(
    #     self,
    #     context,
    #     num_solutions=1,
    #     predict_shape=None,
    #     save_full=False,
    #     oracle_gt_xy=None,
    # ):
    #     del oracle_gt_xy
    #     self.sample_key, key = jr.split(self.sample_key)
    #     sample_keys = jr.split(key, num_solutions)
    #     sample_one = lambda kk: self.sample_one_sol(
    #         self.model,
    #         self.int_beta,
    #         predict_shape,
    #         self.dt0,
    #         self.t1,
    #         context,
    #         save_full,
    #         kk,
    #     )[0]
    #     pred_samples = jax.vmap(sample_one)(sample_keys)
    #     return jnp.mean(pred_samples, axis=0) TODO
