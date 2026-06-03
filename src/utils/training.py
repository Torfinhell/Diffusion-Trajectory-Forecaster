import math
from typing import Any

import equinox as eqx
import jax.random as jr
import optax
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.losses.distill_action import model_feature_dims

PL_TRAINER_KEYS = frozenset(
    {
        "num_epochs",
        "train_epoch_len",
        "val_epoch_len",
        "train_mode",
        "check_val_every_n_epoch",
        "test_epoch_len",
        "log_every_n_steps",
        "enable_profiler",
        "limit_profile_batches",
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


def _resolve_model_dims(hparams: Any) -> dict[str, int]:
    """Resolve denoising and conditioning dimensions from config values."""
    preprocess = hparams.dataset.feat_extract.preprocessing
    current_index = int(preprocess.current_index)
    action_len = int(hparams.trainer.action_len)
    extract_actions = bool(hparams.trainer.extract_actions)
    model_cfg = hparams.model
    model_dim = 91 - current_index - 1
    denoise_steps = max(1, model_dim // action_len) if extract_actions else model_dim
    return {
        "num_agents": hparams.trainer.num_agents,
        "denoise_steps": denoise_steps,
        "past_agent_steps": current_index + 1,
    }


def _configure_model_shapes(model_cfg, dims: dict[str, int]) -> None:
    """Patch model config with runtime-resolved input/output dimensions."""
    num_agents = int(dims["num_agents"])
    denoise_steps = int(dims["denoise_steps"])
    past_agent_steps = int(dims["past_agent_steps"])

    OmegaConf.set_struct(model_cfg, False)
    if getattr(model_cfg, "_target_", "") == "src.models.DiffLinear":
        model_cfg["input_shape"] = [num_agents, past_agent_steps, 2]
        model_cfg["denoise_shape"] = [num_agents, denoise_steps, 2]
        OmegaConf.set_struct(model_cfg, True)
        return

    if getattr(model_cfg, "_target_", "") == "src.models.DiffAttention":
        model_cfg["denoise_shape"] = [num_agents, denoise_steps, 2]
        model_cfg.final_out_dim = denoise_steps * 2
        model_cfg.se_args.time_len = past_agent_steps
        model_cfg.se_args.num_feat = 2
        OmegaConf.set_struct(model_cfg, True)


def build_training_modules(hparams: Any, train_mode: str) -> dict[str, Any]:
    """Instantiate model, loss, optimizer state, and PRNG keys from Hydra config."""
    trainer = hparams.trainer
    grad_clip = trainer.get("grad_clip")
    key = jr.PRNGKey(int(trainer.seed))
    dims = _resolve_model_dims(hparams)
    _configure_model_shapes(hparams.model, dims)
    if train_mode == "distillation":
        _configure_model_shapes(hparams.teacher_model, dims)

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
    optimizer_cfg = hparams.optimizer
    optimizer_args = {}
    if schedule is not None:
        optimizer_args["learning_rate"] = schedule
    optim = clip_optimizer(instantiate(optimizer_cfg, **optimizer_args), grad_clip)
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
