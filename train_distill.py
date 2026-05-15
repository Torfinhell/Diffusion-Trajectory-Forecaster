import math
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import equinox as eqx
import jax.random as jr
import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.losses.distillation_loss import KDLoss, KDProjectors
from src.trainers import BaseTrainerDebug
from src.utils import (
    load_best_checkpoint,
    log_run_metadata,
    process_hparams,
    resolve_scheduler_decay_steps,
)


@hydra.main(version_base=None, config_name="ddpm_attn_distill", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=False)

    logger = None
    if hparams.get("logger", None) is not None:
        logger = instantiate(hparams.logger) if getattr(hparams, "logger", None) else None
    if logger is not None:
        log_run_metadata(logger, hparams)

    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    dm.setup("fit")
    if hparams.trainer.get("train_epoch_len", None) is not None:
        resolve_scheduler_decay_steps(hparams, dm)

    train_epoch_len = hparams.trainer.train_epoch_len
    if isinstance(train_epoch_len, float):
        total_train_batches = len(dm.train_dataloader())
        train_epoch_len = max(1, math.ceil(total_train_batches * train_epoch_len))

    val_epoch_len = hparams.trainer.val_epoch_len
    if isinstance(val_epoch_len, float):
        total_val_batches = len(dm.val_dataloader())
        val_epoch_len = max(1, math.ceil(total_val_batches * val_epoch_len))

    seed = hparams.trainer.seed
    teacher_key, proj_key = jr.split(jr.PRNGKey(seed + 99), 2)

    # Build and load teacher
    teacher = instantiate(hparams.teacher_model, key=teacher_key)
    teacher_ckpt = cfg.distill.teacher_checkpoint
    teacher = eqx.tree_deserialise_leaves(teacher_ckpt, teacher)
    print(f"Loaded teacher from {teacher_ckpt}")

    def _dims_from_cfg(model_cfg):
        return {
            "kv_cond": int(model_cfg.se_args.out_dim),
            "sa": [int(model_cfg.samlp_args.out_dim)] * int(model_cfg.num_sa_mlp),
            "ca": [int(model_cfg.camlp_args.out_dim)] * int(model_cfg.num_camlp),
        }

    projectors = KDProjectors(
        student_dims=_dims_from_cfg(hparams.model),
        teacher_dims=_dims_from_cfg(hparams.teacher_model),
        key=proj_key,
    )

    distill_loss = KDLoss(
        teacher=teacher,
        projectors=projectors,
        lambdas=dict(hparams.distill.lambdas),
        accel_scale=float(hparams.loss.accel_scale),
        yaw_rate_scale=float(hparams.loss.yaw_rate_scale),
    )

    diff_trainer = BaseTrainerDebug(
        seed=seed,
        cfg_metrics=hparams.metrics,
        vis_cfg=hparams.visual,
        model=hparams.model,          
        loss=distill_loss,             
        optimizer=hparams.optimizer,
        scheduler=hparams.get("scheduler", None),
        diffusion_sampler=hparams.diffusion_sampler,
        grad_clip=hparams.trainer.gradient_clip_val,
        trainer_cfg=hparams.trainer,
    )

    check_val_every_n_epoch = hparams.trainer.check_val_every_n_epoch
    val_check_interval = train_epoch_len * check_val_every_n_epoch

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=hparams.trainer.num_epochs,
        logger=logger,
        callbacks=[RichProgressBar(leave=True)],
        enable_progress_bar=True,
        limit_train_batches=train_epoch_len,
        limit_val_batches=val_epoch_len,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        limit_test_batches=hparams.trainer.get("test_epoch_len", 1.0),
        log_every_n_steps=hparams.trainer.get("log_every_n_steps", 1),
    )

    trainer.fit(diff_trainer, dm)

    if bool(hparams.trainer.get("run_test_after_fit", False)):
        if bool(hparams.trainer.get("test_with_best_checkpoint", True)):
            load_best_checkpoint(diff_trainer)
        dm.setup("test")
        trainer.test(diff_trainer, datamodule=dm)


if __name__ == "__main__":
    main()
