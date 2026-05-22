import math
from typing import Any

import equinox as eqx
import jax.random as jr
from hydra.utils import instantiate

from src.losses.distillation_loss import KDProjectors


def resolve_epoch_len(epoch_len: int | float, num_batches: int) -> int:
    """Turn a fractional or absolute batch limit into a step count for Lightning."""
    if isinstance(epoch_len, float):
        return max(1, math.ceil(num_batches * epoch_len))
    return min(num_batches, int(epoch_len))


def model_feature_dims(model_cfg: Any) -> dict[str, int | list[int]]:
    """Attention MLP output dims used to size KD projectors."""
    return {
        "kv_cond": int(model_cfg.se_args.out_dim),
        "sa": [int(model_cfg.samlp_args.out_dim)] * int(model_cfg.num_sa_mlp),
        "ca": [int(model_cfg.camlp_args.out_dim)] * int(model_cfg.num_camlp),
    }


def load_teacher(hparams: Any, seed: int) -> tuple[eqx.Module, KDProjectors]:
    """Instantiate teacher from config, load checkpoint, and build KD projectors."""
    teacher_key, proj_key = jr.split(jr.PRNGKey(seed + 99), 2)
    teacher = instantiate(hparams.teacher_model, key=teacher_key)
    teacher_ckpt = hparams.trainer.get("teacher_checkpoint")
    if teacher_ckpt is None:
        raise ValueError("trainer.teacher_checkpoint must be set for distillation")
    teacher = eqx.tree_deserialise_leaves(teacher_ckpt, teacher)
    projectors = KDProjectors(
        student_dims=model_feature_dims(hparams.model),
        teacher_dims=model_feature_dims(hparams.teacher_model),
        key=proj_key,
    )
    return teacher, projectors
