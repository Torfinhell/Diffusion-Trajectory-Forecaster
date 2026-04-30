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
        diffusion_sampler=None,
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
        self.diffusion_sampler = (
            instantiate(diffusion_sampler)
            if diffusion_sampler is not None
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

    def _update_metrics_for_batch(self, metrics, batch, num_samples=10):
        for sample_idx in range(min(batch["agent_future"].shape[0], num_samples)):
            gt_xy = batch["agent_future"][sample_idx][..., :2]
            future_valid = batch["agent_future_valid"][sample_idx]
            pred_xy = self.sample_one_sol(
                self.model,
                self.diffusion_sampler,
                gt_xy.shape,
                self.model.prepare_conditioning(batch, sample_idx),
                save_full=False,
            )[0]
            metrics.update(
                pred_xy,
                gt_xy,
                batch["agents_coeffs"][sample_idx],
                future_valid,
            )


    def sample_one_sol(
        self,
        model,
        diffusion_sampler,
        data_shape,
        context,
        save_full=False,
        key=None,
    ):
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        step_keys = jr.split(key, diffusion_sampler.num_steps + 1)
        x = jr.normal(step_keys[0], data_shape)
        path = []
        for timestep, step_key in zip(
            range(diffusion_sampler.num_steps - 1, -1, -1), step_keys[1:]
        ):
            timestep_arr = jnp.asarray(timestep, dtype=jnp.int32)
            model_output = model(
                jnp.asarray(timestep, dtype=x.dtype),
                x,
                context,
            )
            x = diffusion_sampler.step(
                step_key,
                model_output,
                timestep_arr,
                x,
            )
            if save_full:
                path.append(x)

        if save_full:
            return jnp.stack(path, axis=0)
        return x[None, ...]

    def training_step(self, batch):
        self._step(batch, "train")
        return None

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        metric_every = max(1, int(self.trainer_cfg.get("val_metric_every_n_epochs", 1)))
        should_run_metrics = (
            batch_idx == 0
            and len(self.metrics_val) > 0
            and ((self.current_epoch + 1) % metric_every == 0 or self.current_epoch == 0)
        )
        if should_run_metrics:
            num_samples = 10
            if self.trainer.sanity_checking:
                num_samples = 1
            self.metrics_val.reset()
            self._update_metrics_for_batch(self.metrics_val, batch, num_samples)
            vals = self.metrics_val.compute()
            log_dict = {
                f"val/{k}": float(jnp.asarray(v))
                for k, v in vals.items()
            }
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch["agent_future"].shape[0],
            )
        return loss

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
        if step_out["train_stats"] is not None:
            for stat_key, stat_value in step_out["train_stats"].items():
                log_metrics[f"{kind}/{stat_key}"] = float(jnp.asarray(stat_value))

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
            grad_fn = eqx.filter_value_and_grad(
                Basetreainer.batch_loss_fn, has_aux=True
            )
            (loss, train_stats), grads = grad_fn(
                model, diffusion_sampler, loss_fn, batch, key
            )
            grad_norm = optax.global_norm(grads)
            updates, opt_state = opt_update(grads, opt_state)
            update_norm = optax.global_norm(updates)
            model = eqx.apply_updates(model, updates)
            param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        else:
            loss, train_stats = Basetreainer.batch_loss_fn(
                model, diffusion_sampler, loss_fn, batch, key
            )
            grad_norm = None
            update_norm = None
            param_norm = None

        #key = jr.split(key, 1)[0]
        return {
            "loss": loss,
            "train_stats": train_stats,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "model": model,
            #"key": key,
            "opt_state": opt_state,
        }

    @staticmethod
    @eqx.filter_jit
    def batch_loss_fn(model, diffusion_sampler, loss_fn, batch, key):
        batch_size = batch["agent_future"].shape[0]
        loss_keys = jr.split(key, batch_size)
        sample_loss_fn = lambda sample, sample_key: loss_fn(
            model, diffusion_sampler, sample, sample_key
        )
        losses, stats = jax.vmap(sample_loss_fn)(batch, loss_keys)
        mean_stats = jax.tree.map(lambda x: jnp.mean(x, axis=0), stats)
        return jnp.mean(losses), mean_stats
