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
        diffusion_scheduler=None,
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
        self.model = instantiate(model, key=key_model)
        self.loss_fn = instantiate(loss)
        self.learning_rate_schedule = (
            instantiate(scheduler) if scheduler is not None else None
        )
        self.diffusion_scheduler = (
            instantiate(diffusion_scheduler)
            if diffusion_scheduler is not None
            else None
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
            diffusion_scheduler=self.diffusion_scheduler,
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
        if step_out["train_stats"] is not None:
            for stat_key, stat_value in step_out["train_stats"].items():
                log_metrics[f"{kind}/{stat_key}"] = float(jnp.asarray(stat_value))

        self.log_dict(
            log_metrics,
            on_step=self.global_step_,
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
        diffusion_scheduler,
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
                Basetreainer.batch_loss_fn, has_aux=True
            )
            (loss, train_stats), grads = grad_fn(
                model, diffusion_scheduler, loss_fn, batch, key
            )
            grad_norm = optax.global_norm(grads)
            updates, opt_state = opt_update(grads, opt_state)
            update_norm = optax.global_norm(updates)
            model = eqx.apply_updates(model, updates)
            param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        else:
            loss, train_stats = Basetreainer.batch_loss_fn(
                model, diffusion_scheduler, loss_fn, batch, key
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
    @eqx.filter_jit
    def batch_loss_fn(model, diffusion_scheduler, loss_fn, batch, key):
        batch_size = batch["agent_future"].shape[0]
        sample_key, losskey = jr.split(key, 3)
        num_steps = diffusion_scheduler.num_steps
        sample_steps = jr.randint(
            key=sample_key, minval=1, maxval=num_steps, shape=(batch_size, num_steps)
        )
        losskey = jr.split(losskey, batch_size)
        sample_loss_fn = partial(loss_fn, model)
        sample_loss_fn = jax.vmap(sample_loss_fn)
        get_model_output = partial(
            Basetreainer.get_model_output,
            model,
            diffusion_scheduler,
        )
        get_model_output = jax.vmap(get_model_output)
        model_output = get_model_output(sample_steps)
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(model_output, **batch))

    @staticmethod
    @eqx.filter_jit
    def get_model_output(model, diffusion_scheduler, timesteps):
        step_fn = jax.vmap(
            partial(
                diffusion_scheduler.step,
            )
        )
