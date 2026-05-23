import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.trainers import (
    BaseProfilerDebug,
    BaseTrainer,
    BaseTrainerDebug,
    BaseTrainerDistillation,
)
from src.utils import (
    build_training_modules,
    load_best_checkpoint,
    log_run_metadata,
    process_hparams,
    resolve_epoch_len,
    resolve_scheduler_decay_steps,
    split_trainer_config,
)


@hydra.main(version_base=None, config_name="ddpm_attn", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=False)
    logger = None
    if hparams.get("logger", None) is not None:
        logger = (
            instantiate(hparams.logger) if getattr(hparams, "logger", None) else None
        )
    if logger is not None:
        log_run_metadata(logger, hparams)

    dm = DiffusionTrackerDataModule(hparams.dataset.data, hparams.dataloaders)
    dm.setup("fit")
    if hparams.trainer.get("train_epoch_len", None) is not None:
        resolve_scheduler_decay_steps(hparams, dm)

    pl_trainer_cfg, module_trainer_cfg = split_trainer_config(hparams.trainer)
    train_mode = pl_trainer_cfg.get("train_mode", "train")
    is_distillation = train_mode == "distillation"

    if train_mode not in ("train", "debug", "profiler", "distillation"):
        raise ValueError(
            f"Unknown train_mode={train_mode!r}; "
            "expected one of train, debug, profiler, distillation"
        )

    modules = build_training_modules(hparams, train_mode)
    trainer_common = dict(
        cfg_metrics=hparams.metrics,
        vis_cfg=hparams.visual,
        **modules,
        **module_trainer_cfg,
    )

    logger_name = logger.name if logger is not None else "default_run"
    jax_profiler_dir = f"./clearml/{logger_name}/jax_profiler"

    if train_mode == "profiler":
        diff_trainer = BaseProfilerDebug(
            log_dir=jax_profiler_dir,
            start_step=pl_trainer_cfg.get("jax_profiler_start_step", 2),
            num_steps=pl_trainer_cfg.get("jax_profiler_num_steps", 3),
            **trainer_common,
        )
    elif is_distillation:
        diff_trainer = BaseTrainerDistillation(**trainer_common)
    elif train_mode == "debug":
        diff_trainer = BaseTrainerDebug(**trainer_common)
    else:
        diff_trainer = BaseTrainer(**trainer_common)

    train_epoch_len = resolve_epoch_len(
        pl_trainer_cfg["train_epoch_len"], len(dm.train_dataloader())
    )
    val_epoch_len = resolve_epoch_len(
        pl_trainer_cfg["val_epoch_len"], len(dm.val_dataloader())
    )

    trainer_kwargs = dict(
        accelerator="gpu",
        max_epochs=pl_trainer_cfg["num_epochs"],
        logger=logger,
        callbacks=[RichProgressBar(leave=True)],
        enable_progress_bar=True,
        limit_train_batches=train_epoch_len,
        limit_val_batches=val_epoch_len,
    )
    if is_distillation:
        check_val_every_n_epoch = int(pl_trainer_cfg["check_val_every_n_epoch"])
        trainer_kwargs["val_check_interval"] = train_epoch_len * check_val_every_n_epoch
        trainer_kwargs["check_val_every_n_epoch"] = None
        trainer_kwargs["limit_test_batches"] = pl_trainer_cfg.get("test_epoch_len", 1.0)
        trainer_kwargs["log_every_n_steps"] = pl_trainer_cfg.get("log_every_n_steps", 1)
    else:
        trainer_kwargs["check_val_every_n_epoch"] = pl_trainer_cfg[
            "check_val_every_n_epoch"
        ]

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(diff_trainer, dm)

    if bool(pl_trainer_cfg.get("run_test_after_fit", False)):
        if bool(pl_trainer_cfg.get("test_with_best_checkpoint", True)):
            load_best_checkpoint(diff_trainer)
        dm.setup("test")
        trainer.test(diff_trainer, datamodule=dm)


if __name__ == "__main__":
    main()
