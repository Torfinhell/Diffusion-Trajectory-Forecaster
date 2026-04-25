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
from src.utils import load_best_checkpoint, log_model_artifact
from utils.eval import on_test_epoch_end as run_test_epoch_end
from utils.eval import on_train_epoch_end as run_train_epoch_end
from utils.eval import on_validation_epoch_end as run_validation_epoch_end
from utils.eval import update_metrics_for_batch
from utils.logging import log_training_step_stats
from utils.proxy import compute_proxy_batch_loss


class BaseDiffusionModel(L.LightningModule):
    CHECKPOINT_ROOT = Path("checkpoints")

    def __init__(
        self,
        seed,
        load_best_checkpoint,
        cfg_metrics,
        vis_cfg,
        grad_clip=None,
        trainer_cfg=None,
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
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        if self.grad_clip is not None and self.grad_clip <= 0:
            self.grad_clip = None
        self.load_best_checkpoint_flag = load_best_checkpoint
        self.samples = 10
        self.global_step_ = 0  # TODO
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
        self.proxy_val_cfg = self.trainer_cfg.get("proxy_val_loss", {})
        default_best_metric = (
            "val_proxy_loss" if self.proxy_val_cfg.get("enabled", False) else "val_loss"
        )
        self.best_checkpoint_metric = str(
            self.trainer_cfg.get("best_checkpoint_metric", default_best_metric)
        )
        self.best_checkpoint_mode = str(
            self.trainer_cfg.get("best_checkpoint_mode", "min")
        ).lower()
        self.best_checkpoint_score = (
            float("inf") if self.best_checkpoint_mode == "min" else float("-inf")
        )
        self.best_checkpoint_epoch = None
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
        if scheduler_cfg is not None:
            name = str(scheduler_cfg.get("name", "none")).lower()

        if scheduler_cfg is None or name in {"none", "off", "disabled"}:
            learning_rate = float(base_lr)
            self.learning_rate_schedule = learning_rate
            return learning_rate

        decay_steps = int(scheduler_cfg.get("decay_steps", 0))
        if decay_steps <= 0:
            raise ValueError(
                f"lr_scheduler.decay_steps must be > 0 for scheduler '{name}'"
            )

        end_lr = float(scheduler_cfg.get("end_lr", 0.0))

        if name == "cosine":
            alpha = end_lr / float(base_lr) if base_lr > 0 else 0.0
            learning_rate = optax.cosine_decay_schedule(
                init_value=float(base_lr),
                decay_steps=decay_steps,
                alpha=alpha,
            )
            self.learning_rate_schedule = learning_rate
            return learning_rate

        if name == "linear":
            learning_rate = optax.linear_schedule(
                init_value=float(base_lr),
                end_value=end_lr,
                transition_steps=decay_steps,
            )
            self.learning_rate_schedule = learning_rate
            return learning_rate

        raise ValueError(f"Unsupported lr scheduler '{name}'")

    def clip_optimizer(self, optimizer):
        transforms = []
        if self.grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip))
        transforms.append(optimizer)
        return optax.chain(*transforms)

    def training_step(self, batch):
        with jax.profiler.StepTraceAnnotation("train", step_num=int(self.global_step_)):
            (
                value,
                train_stats,
                grad_norm,
                update_norm,
                param_norm,
                self.model,
                self.train_key,
                self.opt_state,
            ) = BaseDiffusionModel.make_step(
                self.model,
                self.batch_loss,
                self.int_beta,
                batch,
                self.t1,
                self.train_key,
                self.opt_state,
                self.optim.update,
            )
            self.train_loss_tracker.update("train_loss", jnp.asarray(value))
            log_training_step_stats(
                self,
                grad_norm=grad_norm,
                update_norm=update_norm,
                param_norm=param_norm,
                train_stats=train_stats,
            )
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
            value = self.compute_batch_loss(batch, val_key)
            self.val_loss_tracker.update("val_loss", jnp.asarray(value))
            if bool(self.proxy_val_cfg.get("enabled", False)):
                self.loader_key, proxy_key = jr.split(self.loader_key)
                proxy_value = compute_proxy_batch_loss(self, batch, proxy_key)
                self.val_loss_tracker.update("val_proxy_loss", jnp.asarray(proxy_value))
            # collect some first batches to compute metrics on
            if (
                self._should_run_metrics_this_epoch("val")
                and len(self._val_batches_for_metrics) < self.val_metric_batches
            ):
                self._val_batches_for_metrics.append(batch)

    def test_step(self, batch):
        with jax.profiler.StepTraceAnnotation("test", step_num=int(self.current_epoch)):
            self._update_metrics_for_batch(
                self.metrics_test,
                batch,
                return_first_prediction=False,
            )

    @staticmethod
    @eqx.filter_jit
    def make_step(
        model,
        batch_loss,
        int_beta,
        batch,
        t1,
        key,
        opt_state,
        opt_update,
    ):
        loss_fn = eqx.filter_value_and_grad(batch_loss, has_aux=True)
        (loss, train_stats), grads = loss_fn(model, int_beta, batch, t1, key)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = opt_update(grads, opt_state)
        update_norm = optax.global_norm(updates)
        model = eqx.apply_updates(model, updates)
        param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        key = jr.split(key, 1)[0]
        return (
            loss,
            train_stats,
            grad_norm,
            update_norm,
            param_norm,
            model,
            key,
            opt_state,
        )

    @staticmethod
    def batch_loss_fn(model, loss_fn, int_beta, batch, t, key):
        # debug class overrides it
        batch_size = batch["agent_future"].shape[0]
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        t_min = min(0.1, float(t) * 0.5)
        t = jr.uniform(tkey, (batch_size,), minval=t_min, maxval=t)
        loss_fn = partial(loss_fn, model, int_beta)
        loss_fn = jax.vmap(loss_fn)
        losses = loss_fn(batch, t, losskey)
        return jnp.mean(losses)

    def sample_multiple_sol(
        self,
        context,
        num_solutions=1,
        predict_shape=None,
        save_full=False,
    ):
        """
        Sample trajectory predictions and average over multiple stochastic draws.
        """
        self.sample_key, key = jr.split(self.sample_key)
        sample_keys = jr.split(key, num_solutions)
        # The sampler returns one trajectory per random seed, then averages them
        # into a single prediction for downstream metrics/logging.
        sample_one = lambda kk: self.sample_one_sol(
            model=self.model,
            int_beta=self.int_beta,
            data_shape=predict_shape,
            dt0=self.dt0,
            t1=self.t1,
            context=context,
            save_full=save_full,
            key=kk,
        )[0]
        pred_samples = jax.vmap(sample_one)(sample_keys)
        return jnp.mean(pred_samples, axis=0)

    @eqx.filter_jit
    def sample_one_sol(
        self, model, int_beta, data_shape, dt0, t1, context, save_full=False, key=None
    ):
        raise NotImplementedError("Should implement Sample one sol in derived class")

    def single_loss(model, int_beta, batch, t, key):
        raise NotADirectoryError("Should Implement loss in derived class")

    def _should_run_metrics_this_epoch(self, split):
        every_n = max(1, int(self.trainer_cfg.get(f"{split}_metric_every_n_epochs", 1)))
        return (int(self.current_epoch) + 1) % every_n == 0

    def on_fit_end(self):
        log_model_artifact(self)

    def on_train_epoch_end(self) -> None:
        return run_train_epoch_end(self)

    def on_validation_epoch_start(self) -> None:
        self.val_loss_tracker.reset()

    def on_validation_epoch_end(self) -> None:
        return run_validation_epoch_end(self)

    def on_test_epoch_end(self) -> None:
        return run_test_epoch_end(self)

    def _update_metrics_for_batch(self, metrics, batch, return_first_prediction=False):
        # debug subclass overrides it
        return update_metrics_for_batch(
            self,
            metrics,
            batch,
            return_first_prediction=return_first_prediction,
        )
