import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.trainers import BaseTrainer
from src.utils import (
    build_training_modules,
    load_best_checkpoint,
    log_run_metadata,
    process_hparams,
    resolve_epoch_len,
    split_trainer_config,
)


@hydra.main(version_base=None, config_name="inference", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=False)
    logger = instantiate(hparams.logger) if hparams.get("logger") else None
    if logger is not None:
        log_run_metadata(logger, hparams)

    dm = DiffusionTrackerDataModule(hparams.dataset.data, hparams.dataloaders)
    dm.setup("test")

    pl_trainer_cfg, module_trainer_cfg = split_trainer_config(hparams.trainer)
    modules = build_training_modules(
        hparams, pl_trainer_cfg.get("train_mode", "inference")
    )
    diff_trainer = BaseTrainer(
        cfg_metrics=hparams.metrics,
        vis_cfg=hparams.visual,
        **modules,
        **module_trainer_cfg,
    )
    if bool(pl_trainer_cfg.get("test_with_best_checkpoint", True)):
        load_best_checkpoint(diff_trainer)

    test_epoch_len = resolve_epoch_len(
        pl_trainer_cfg.get("test_epoch_len", 1.0),
        len(dm.test_dataloader()),
    )
    trainer = Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=[RichProgressBar(leave=True)],
        enable_progress_bar=True,
        limit_test_batches=test_epoch_len,
    )
    trainer.test(diff_trainer, datamodule=dm)


if __name__ == "__main__":
    main()
