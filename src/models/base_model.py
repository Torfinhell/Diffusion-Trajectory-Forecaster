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

from src.metrics import MetricCollection
from src.models.base_model_eval import (
    on_train_epoch_end as run_train_epoch_end,
    on_validation_epoch_end as run_validation_epoch_end,
    update_metrics_for_batch,
)
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
        self.samples = 10
        self.global_step_ = 0  # TODO
        self.metrics_train = MetricCollection(
            [instantiate(m) for m in self.metrics.train]
        )
        self.metrics_val = MetricCollection([instantiate(m) for m in self.metrics.val])
        self._train_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        self._train_loss_count = 0
        self._val_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        self._val_loss_count = 0
        self._val_proxy_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        self._val_proxy_loss_count = 0
        self.val_metric_batches = int(self.trainer_cfg.get("val_metric_batches", 3))
        self._val_batches_for_metrics = []
        self.train_metric_batches = int(
            self.trainer_cfg.get("train_metric_batches", 3)
        )
        self._train_batches_for_metrics = []
        self.metrics_prefix = "Val"
        self.prediction_target = str(prediction_target).lower()
        if self.prediction_target not in {"x0", "epsilon", "score"}:
            raise ValueError(
                f"Unsupported prediction_target '{prediction_target}'. "
                "Use one of: x0, epsilon, score."
            )
        self.proxy_val_cfg = self.trainer_cfg.get("proxy_val_loss", {})
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
            self._train_loss_sum = self._train_loss_sum + jnp.asarray(value)
            self._train_loss_count += 1
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
            self._val_loss_sum = self._val_loss_sum + jnp.asarray(value)
            self._val_loss_count += 1
            if self._proxy_val_enabled():
                self.loader_key, proxy_key = jr.split(self.loader_key)
                proxy_value = self.compute_proxy_batch_loss(batch, proxy_key)
                self._val_proxy_loss_sum = self._val_proxy_loss_sum + jnp.asarray(
                    proxy_value
                )
                self._val_proxy_loss_count += 1
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

    def _sampling_t0(self):
        # Subclasses can override this when sampling should start from a
        # different noise level than the model default.
        if hasattr(self, "t0"):
            return float(self.t0)
        return 1e-3

    def _oracle_enabled(self, key):
        del key
        return False

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
        del use_oracle
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
    def batch_loss_fn_fixed_t(model, weight, int_beta, prediction_target, batch, t, key):
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

    def on_train_epoch_start(self) -> None:
        self._train_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        self._train_loss_count = 0

    def on_train_epoch_end(self) -> None:
        return run_train_epoch_end(self)

    def on_validation_epoch_start(self) -> None:
        self._val_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        self._val_loss_count = 0
        self._val_proxy_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        self._val_proxy_loss_count = 0

    def on_validation_epoch_end(self) -> None:
        self._save_local_checkpoint()
        return run_validation_epoch_end(self)

    def _save_local_checkpoint(self) -> None:
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(self.CHECKPOINT_PATH, self.model)

    def _proxy_val_enabled(self) -> bool:
        return bool(self.proxy_val_cfg.get("enabled", False))

    def _proxy_t_values(self):
        step_stride = max(1, int(self.proxy_val_cfg.get("step_stride", 5)))
        include_last = bool(self.proxy_val_cfg.get("include_last", True))
        t0 = float(self._sampling_t0())
        t1 = float(self.t1)
        dt0 = abs(float(getattr(self, "dt0", 0.01)))
        num_steps = max(1, int(np.ceil((t1 - t0) / dt0)))
        ts = np.linspace(t0, t1, num_steps + 1, dtype=np.float32)[1:]
        selected = ts[step_stride - 1 :: step_stride]
        if include_last and (len(selected) == 0 or selected[-1] != ts[-1]):
            selected = np.concatenate([selected, ts[-1:]])
        return jnp.asarray(selected, dtype=jnp.float32)

    def compute_proxy_batch_loss(self, batch, key):
        t_values = self._proxy_t_values()
        keys = jr.split(key, int(t_values.shape[0]))

        def loss_at_t(t, loss_key):
            return self.batch_loss_fn_fixed_t(
                self.model,
                self.weight,
                self.int_beta,
                self.prediction_target,
                batch,
                t,
                loss_key,
            )

        losses = jax.vmap(loss_at_t)(t_values, keys)
        return jnp.mean(losses)

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

    def _render_prediction_image(self, viz_state, pred_xy_world, plot_kwargs):
        return plot_simulator_state(
            viz_state,
            batch_idx=0,
            pred_xy=pred_xy_world,
            **plot_kwargs,
        )

    def _update_metrics_for_batch(
        self, metrics, batch, return_first_prediction=False
    ):
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
            oracle_gt_xy=oracle_gt_xy,
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
        oracle_gt_xy=None,
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
