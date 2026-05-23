import math
from typing import Any

import equinox as eqx
import jax.random as jr
import optax
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.losses.distillation_loss import model_feature_dims

PL_TRAINER_KEYS = frozenset(
    {
        "num_epochs",
        "train_epoch_len",
        "val_epoch_len",
        "train_mode",
        "check_val_every_n_epoch",
        "test_epoch_len",
        "log_every_n_steps",
        "jax_profiler_start_step",
        "jax_profiler_num_steps",
        "teacher_checkpoint",
        "print_hparams",
        "logging",
    }
)


def split_trainer_config(trainer_cfg):
    cfg = dict(OmegaConf.to_container(trainer_cfg, resolve=True))
    pl_cfg = {k: cfg.pop(k) for k in PL_TRAINER_KEYS if k in cfg}
    return pl_cfg, cfg


def resolve_epoch_len(epoch_len: int | float, num_batches: int) -> int:
    """Turn a fractional or absolute batch limit into a step count for Lightning."""
    if isinstance(epoch_len, float):
        return max(1, math.ceil(num_batches * epoch_len))
    return min(num_batches, int(epoch_len))


def clip_optimizer(optimizer, grad_clip: float | None):
    transforms = []
    if grad_clip is not None and float(grad_clip) > 0:
        transforms.append(optax.clip_by_global_norm(float(grad_clip)))
    transforms.append(optimizer)
    return optax.chain(*transforms)


def build_training_modules(hparams: Any, train_mode: str) -> dict[str, Any]:
    """Instantiate model, loss, optimizer state, and PRNG keys from Hydra config."""
    trainer = hparams.trainer
    grad_clip = trainer.get("grad_clip")
    key = jr.PRNGKey(int(trainer.seed))

    if train_mode == "distillation":
        key, key_model, key_teacher, key_proj, train_key, loader_key, sample_key = (
            jr.split(key, 7)
        )
        teacher_ckpt = trainer.get("teacher_checkpoint")
        if teacher_ckpt is None:
            raise ValueError("trainer.teacher_checkpoint must be set for distillation")
        teacher = instantiate(hparams.teacher_model, key=key_teacher)
        teacher = eqx.tree_deserialise_leaves(teacher_ckpt, teacher)
        print(f"Loaded teacher from {teacher_ckpt}")
        model = instantiate(hparams.model, key=key_model)
        loss_fn = instantiate(
            hparams.loss,
            teacher=teacher,
            student_dims=model_feature_dims(hparams.model),
            teacher_dims=model_feature_dims(hparams.teacher_model),
            key=key_proj,
        )
        trainable = eqx.filter((model, loss_fn.projectors), eqx.is_inexact_array)
    else:
        key, key_model, train_key, loader_key, sample_key = jr.split(key, 5)
        model = instantiate(hparams.model, key=key_model)
        loss_fn = instantiate(hparams.loss)
        trainable = eqx.filter(model, eqx.is_inexact_array)

    diffusion_sampler = instantiate(hparams.diffusion_sampler)
    schedule = instantiate(hparams.scheduler) if hparams.get("scheduler") else None
    optimizer_args = {}
    if schedule is not None:
        optimizer_args["learning_rate"] = schedule
    optim = clip_optimizer(instantiate(hparams.optimizer, **optimizer_args), grad_clip)
    opt_state = optim.init(trainable)

    return dict(
        model=model,
        loss_fn=loss_fn,
        optim=optim,
        opt_state=opt_state,
        diffusion_sampler=diffusion_sampler,
        key=key,
        train_key=train_key,
        loader_key=loader_key,
        sample_key=sample_key,
    )
