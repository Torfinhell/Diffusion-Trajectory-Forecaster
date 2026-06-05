import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.trainers import BaseTrainer, BaseTrainerDistillation
from src.utils import (
    JaxProfilerCallback,
    build_training_modules,
    load_best_checkpoint,
    log_run_metadata,
    process_hparams,
    resolve_epoch_len,
    resolve_scheduler_decay_steps,
    split_trainer_config,
)


def build_callbacks(pl_trainer_cfg, logger_name: str):
    callbacks = [RichProgressBar(leave=True)]
    if pl_trainer_cfg.get("enable_profiler", False):
        log_dir = f"./clearml/{logger_name}/jax_profiler"
        callbacks.append(
            JaxProfilerCallback(
                log_dir=log_dir,
                limit_profile_batches=int(
                    pl_trainer_cfg.get("limit_profile_batches", 3)
                ),
            )
        )
    return callbacks


@hydra.main(version_base=None, config_name="ddpm_attn", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=False)
    logger = instantiate(hparams.logger) if hparams.get("logger") else None
    if logger is not None:
        log_run_metadata(logger, hparams)

    dm = DiffusionTrackerDataModule(hparams.dataset.data, hparams.dataloaders)
    dm.setup("fit")
    if hparams.trainer.get("train_epoch_len", None) is not None:
        resolve_scheduler_decay_steps(hparams, dm)

    pl_trainer_cfg, module_trainer_cfg = split_trainer_config(hparams.trainer)
    train_mode = pl_trainer_cfg.get("train_mode", "train")
    if train_mode not in ("train", "debug", "distillation"):
        raise ValueError(
            f"Unknown train_mode={train_mode!r}; expected train, debug, or distillation"
        )

    modules = build_training_modules(hparams, train_mode)
    if train_mode == "debug":
        module_trainer_cfg = {**module_trainer_cfg, "debug": True}

    trainer_common = dict(
        cfg_metrics=hparams.metrics,
        vis_cfg=hparams.visual,
        **modules,
        **module_trainer_cfg,
    )
    diff_trainer = (
        BaseTrainerDistillation(**trainer_common)
        if train_mode == "distillation"
        else BaseTrainer(**trainer_common)
    )

    train_epoch_len = resolve_epoch_len(
        pl_trainer_cfg["train_epoch_len"], len(dm.train_dataloader())
    )
    val_epoch_len = resolve_epoch_len(
        pl_trainer_cfg["val_epoch_len"], len(dm.val_dataloader())
    )
    logger_name = logger.name if logger is not None else "default_run"
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=pl_trainer_cfg["num_epochs"],
        logger=logger,
        callbacks=build_callbacks(pl_trainer_cfg, logger_name),
        enable_progress_bar=True,
        limit_train_batches=train_epoch_len,
        limit_val_batches=val_epoch_len,
        check_val_every_n_epoch=pl_trainer_cfg.get("check_val_every_n_epoch", 1),
        limit_test_batches=pl_trainer_cfg.get("test_epoch_len", 1.0),
        log_every_n_steps=pl_trainer_cfg.get("log_every_n_steps", 1),
    )
    trainer.fit(diff_trainer, dm)
    if bool(pl_trainer_cfg.get("run_test_after_fit", False)):
        if bool(pl_trainer_cfg.get("test_with_best_checkpoint", True)):
            load_best_checkpoint(diff_trainer)
        dm.setup("test")
        trainer.test(diff_trainer, datamodule=dm)


if __name__ == "__main__":
    main()
