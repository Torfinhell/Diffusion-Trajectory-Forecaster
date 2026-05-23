import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from src.trainers.base_trainer import BaseTrainer
from src.utils import (
    load_best_checkpoint,
    log_model_artifact,
    maybe_save_best_checkpoint,
)


class BaseTrainerDebug(BaseTrainer):
    """BaseTrainer with per-term loss stats logged and checkpoint hooks enabled."""

    def _init_trainer_state(self, cfg_metrics, vis_cfg, trainer_cfg):
        trainer_cfg.pop("loss_returns_stats", None)
        super()._init_trainer_state(cfg_metrics, vis_cfg, trainer_cfg)
        self.loss_returns_stats = True

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = {}
        for attr in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
            metrics.update(getattr(self.trainer, attr, None) or {})
        maybe_save_best_checkpoint(self, metrics)

    def on_fit_end(self):
        if bool(self.trainer_cfg.get("load_best_checkpoint", False)):
            load_best_checkpoint(self)
        log_model_artifact(self)

    def _should_run_metrics(self, split: str) -> bool:
        metrics = self.metrics_train if split == "train" else self.metrics_val
        if metrics is None or len(metrics) == 0:
            return False
        every = max(1, int(self.trainer_cfg.get(f"{split}_metric_every_n_epochs", 1)))
        return (self.current_epoch + 1) % every == 0

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
            self._apply_step_updates(step_out)

        log_output = {
            f"{kind}/{key}": float(jnp.asarray(value))
            for key, value in step_out.items()
            if key not in {"model", "opt_state", "projectors"} and value is not None
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

    @staticmethod
    @eqx.filter_jit
    def batch_loss_fn(model, diffusion_sampler, loss_fn, batch, key):
        batch = {name: value for name, value in batch.items() if name != "scenario"}
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        loss_keys = jr.split(key, batch_size)

        def mapped_fn(single_sample_dict, single_key):
            loss, stats = loss_fn(
                model=model,
                diffusion_sampler=diffusion_sampler,
                key=single_key,
                debug=True,
                **single_sample_dict,
            )
            return loss, stats

        losses, stats = jax.vmap(mapped_fn)(batch, loss_keys)
        mean_stats = jax.tree.map(lambda x: jnp.mean(x, axis=0), stats)
        return jnp.mean(losses, axis=0), mean_stats
