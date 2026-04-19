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
from src.models.base_model_eval import on_train_epoch_end as run_train_epoch_end
from src.models.base_model_eval import (
    on_test_epoch_end as run_test_epoch_end,
    on_validation_epoch_end as run_validation_epoch_end,
)
from src.models.base_model_eval import update_metrics_for_batch
from src.models.base_model_proxy import compute_proxy_batch_loss
from src.utils import build_checkpoint_run_directory


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
        self.metrics_test = MetricCollection(
            [instantiate(m) for m in self.metrics.val]
        )
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

    def build_optimizer(self, learning_rate):
        transforms = []
        if self.grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip))
        transforms.append(optax.adam(learning_rate))
        return optax.chain(*transforms)

    def training_step(self, batch):
        with jax.profiler.StepTraceAnnotation("train", step_num=int(self.global_step_)):
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

    def _sampling_t0(self):
        # Override hook for subclasses that need a different sampling start
        if hasattr(self, "t0"):
            return float(self.t0)
        return 1e-3

    def _oracle_enabled(self, key):
        # Override hook for debug subclass
        del key
        return False

    def _oracle_sampling_mode(self):
        # Override hook for debug subclass
        return "exact"

    def _should_run_metrics_this_epoch(self, split):
        every_n = max(1, int(self.trainer_cfg.get(f"{split}_metric_every_n_epochs", 1)))
        return (int(self.current_epoch) + 1) % every_n == 0

    def _upload_artifact(self, name, path, metadata=None):
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "upload_artifact"):
            return
        logger.upload_artifact(name=name, path=path, metadata=metadata)

    def _checkpoint_run_dir(self) -> Path:
        logger = getattr(self, "logger", None)
        return build_checkpoint_run_directory(self.CHECKPOINT_ROOT, logger)

    def _best_checkpoint_path(self) -> Path:
        return self._checkpoint_run_dir() / "best.eqx"

    def _maybe_save_best_checkpoint(self, metrics) -> None:
        metric_value = metrics.get(self.best_checkpoint_metric)
        if metric_value is None:
            return

        score = float(jnp.asarray(metric_value))
        improved = (
            score < self.best_checkpoint_score
            if self.best_checkpoint_mode == "min"
            else score > self.best_checkpoint_score
        )
        if not improved:
            return

        checkpoint_path = self._best_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(checkpoint_path, self.model)
        self.best_checkpoint_score = score
        self.best_checkpoint_epoch = int(self.current_epoch)
        print(
            f"Saved best checkpoint to {checkpoint_path} "
            f"({self.best_checkpoint_metric}={score:.6f})"
        )

    @staticmethod
    def _alpha_sigma(int_beta, t):
        alpha = jnp.exp(-0.5 * int_beta(t))
        sigma = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5))
        return alpha, sigma

    def compute_batch_loss(self, batch, key):
        # debug class overrides it
        return self.batch_loss_fn(
            self.model,
            self.weight,
            self.int_beta,
            self.prediction_target,
            batch,
            self.t1,
            key,
        )

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
    def batch_loss_fn_fixed_t(
        model, weight, int_beta, prediction_target, batch, t, key
    ):
        batch_size = batch["gt_xy"].shape[0]
        losskey = jr.split(key, batch_size)
        t = jnp.full((batch_size,), t, dtype=jnp.float32)
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
        err = (model(t, y, batch["context"]) - batch["gt_xy"]) ** 2
        loss = (err * batch["gt_xy_mask"]).sum() / jnp.maximum(
            batch["gt_xy_mask"].sum(), 1.0
        )
        return loss

    def on_fit_start(self) -> None:
        self.CHECKPOINT_ROOT.mkdir(exist_ok=True)
        print(self.hparams)
        if self.load_best_checkpoint_flag:
            try:
                self.load_best_checkpoint()
            except:
                print("Didnt load weights")

    def on_fit_end(self):
        self._log_model_artifact()

    def on_train_epoch_start(self) -> None:
        self.train_loss_tracker.reset()

    def on_train_epoch_end(self) -> None:
        return run_train_epoch_end(self)

    def on_validation_epoch_start(self) -> None:
        self.val_loss_tracker.reset()

    def on_validation_epoch_end(self) -> None:
        return run_validation_epoch_end(self)

    def on_test_epoch_start(self) -> None:
        self.metrics_test.reset()

    def on_test_epoch_end(self) -> None:
        return run_test_epoch_end(self)

    def _log_model_artifact(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        checkpoint_path = self._best_checkpoint_path()
        if not checkpoint_path.exists():
            return

        artifact_name = f"{getattr(logger, 'name', 'model')}-best-model"
        version = getattr(logger, "version", None)
        if version:
            artifact_name = f"{artifact_name}-{version}"

        self._upload_artifact(
            name=artifact_name,
            path=checkpoint_path,
            metadata={
                "epoch": self.best_checkpoint_epoch,
                "global_step": int(self.global_step_),
                "monitor_metric": self.best_checkpoint_metric,
                "monitor_mode": self.best_checkpoint_mode,
                "monitor_score": self.best_checkpoint_score,
            },
        )

    def load_best_checkpoint(self) -> bool:
        checkpoint_path = self._best_checkpoint_path()
        if not checkpoint_path.exists():
            return False
        self.model = eqx.tree_deserialise_leaves(checkpoint_path, self.model)
        print(f"Loaded best checkpoint from {checkpoint_path}")
        return True


    def _update_metrics_for_batch(self, metrics, batch, return_first_prediction=False):
        # debug subclass overrides it
        return update_metrics_for_batch(
            self,
            metrics,
            batch,
            return_first_prediction=return_first_prediction,
        )

    def sample_multiple_sol(
        self,
        context,
        num_solutions=1,
        predict_shape=None,
        save_full=False,
        oracle_gt_xy=None,
    ):
        """
        Sample trajectory predictions and average over multiple stochastic draws.
        """
        del oracle_gt_xy
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
        self,
        model,
        int_beta,
        data_shape,
        dt0,
        t1,
        context,
        save_full=False,
        key=None,
    ):
        """
        Return full reverse trajectory states sampled at `num_solutions` times.
        Output shape: [num_solutions, data_shape]
        """
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        t0 = self._sampling_t0()
        y1 = jr.normal(key, data_shape)
        num_steps = max(1, int(np.ceil((t1 - t0) / abs(dt0))))
        ts = jnp.linspace(t1, t0, num_steps + 1)
        x = y1
        path = []
        for step_idx in range(num_steps):
            t_cur = ts[step_idx]
            t_next = ts[step_idx + 1]
            alpha_cur, sigma_cur = self._alpha_sigma(int_beta, t_cur)
            alpha_next, sigma_next = self._alpha_sigma(int_beta, t_next)
            pred = model(t_cur, x, context)
            eps_pred = (x - alpha_cur * pred) / jnp.maximum(sigma_cur, 1e-5)
            x = alpha_next * pred + sigma_next * eps_pred
            if save_full:
                path.append(x)
        if save_full:
            return jnp.stack(path, axis=0)
        return x[None, ...]
