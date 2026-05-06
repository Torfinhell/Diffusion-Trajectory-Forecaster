from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytorch_lightning as L
from hydra.utils import instantiate


class BaseProfilerDebug(L.LightningModule):
    CHECKPOINT_ROOT = Path("checkpoints")

    def __init__(
        self,
        seed,
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
        self.key, key_model, self.train_key, self.loader_key = jax.random.split(
            self.key, 4
        )
        self.trainer_cfg = trainer_cfg or {}
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        if self.grad_clip is not None and self.grad_clip <= 0:
            self.grad_clip = None
        self.global_step_ = 0
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

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
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
                BaseProfilerDebug.batch_loss_fn, has_aux=True
            )
            loss, grads = grad_fn(model, diffusion_sampler, loss_fn, batch, key)
            grad_norm = optax.global_norm(grads)
            updates, opt_state = opt_update(grads, opt_state)
            update_norm = optax.global_norm(updates)
            model = eqx.apply_updates(model, updates)
            param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        else:
            loss = BaseProfilerDebug.batch_loss_fn(
                model, diffusion_sampler, loss_fn, batch, key
            )
            grad_norm = None
            update_norm = None
            param_norm = None
        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "model": model,
            "opt_state": opt_state,
        }

    @staticmethod
    @eqx.filter_jit
    def batch_loss_fn(model, diffusion_sampler, loss_fn, batch, key):
        batch = {name: value for name, value in batch.items() if name != "scenario"}
        batch_size = batch["agent_future"].shape[0]
        loss_keys = jr.split(key, batch_size)
        sample_loss_fn = lambda sample, sample_key: loss_fn(
            model, diffusion_sampler, sample, sample_key
        )
        losses = jax.vmap(sample_loss_fn)(batch, loss_keys)
        return jnp.mean(losses)
