from functools import partial
from pathlib import Path

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
        # self.prediction_target = str(prediction_target).lower()
        # if self.prediction_target not in {"x0", "epsilon", "score"}:
        #     raise ValueError(
        #         f"Unsupported prediction_target '{prediction_target}'. "
        #         "Use one of: x0, epsilon, score."
        #     )
        # self.int_beta = lambda t: t
        # self.t0 = float(t0)
        # self.t1 = float(t1)
        # self.dt0 = float(dt0)
        # self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
        # TODO uncomment for now but later need to use diffusion sampler
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
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "val")

    def test_step(self, batch):
        return self._step(batch, "test")

    def _step(self, batch, kind):
        is_train = kind == "train"
        if is_train:
            step_key, self.train_key = jr.split(self.train_key)
        else:
            step_key, self.loader_key = jr.split(self.loader_key)

        step_out = self.make_step(
            model=self.model,
            loss_fn=self.loss_fn,
            int_beta=self.int_beta,
            batch=batch,
            timesteps=self.t1,
            key=step_key,
            train=is_train,
            opt_state=self.opt_state if is_train else None,
            opt_update=self.optim.update if is_train else None,
        )

        if is_train:
            self.model = step_out["model"]
            self.opt_state = step_out["opt_state"]

        log_metrics = {f"{kind}/loss_step": float(jnp.asarray(step_out["loss"]))}
        if step_out.get("grad_norm") is not None:
            log_metrics[f"{kind}/grad_norm"] = float(jnp.asarray(step_out["grad_norm"]))
        if step_out.get("update_norm") is not None:
            log_metrics[f"{kind}/update_norm"] = float(
                jnp.asarray(step_out["update_norm"])
            )
        if step_out.get("param_norm") is not None:
            log_metrics[f"{kind}/param_norm"] = float(
                jnp.asarray(step_out["param_norm"])
            )
        if step_out["train_stats"] is not None:
            for stat_key, stat_value in step_out["train_stats"].items():
                log_metrics[f"{kind}/{stat_key}"] = float(jnp.asarray(stat_value))

        self.log_dict(
            log_metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        if is_train:
            self.global_step_ += 1

        return step_out["loss"]

    @staticmethod
    @eqx.filter_jit
    def make_step(
        model,
        loss_fn,
        int_beta,
        batch,
        timesteps,
        key,
        train,
        opt_state=None,
        opt_update=None,
    ):
        if train:
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
        else:
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
