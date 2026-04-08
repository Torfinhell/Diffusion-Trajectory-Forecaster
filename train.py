import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.utils import log_run_metadata, process_hparams


@hydra.main(version_base=None, config_name="ddpm_baseline", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=True)
    logger = instantiate(hparams.logger)
    profiler = None
    if cfg.trainer.enable_profiler:
        profiler = PyTorchProfiler(
            dirpath=f"./clearml/{logger.name}/profiler_logs", filename="perf_trace"
        )
    log_run_metadata(logger, hparams)
    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    progress_bar = RichProgressBar(leave=True)
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=hparams.trainer.num_epochs,
        logger=logger,
        callbacks=[progress_bar],
        log_every_n_steps=hparams.trainer.log_every_n_steps,
        enable_progress_bar=True,
        limit_train_batches=hparams.trainer.train_epoch_len,
        limit_val_batches=hparams.trainer.val_epoch_len,
        check_val_every_n_epoch=hparams.trainer.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=cfg.trainer.generate_every_epoch,
        profiler=profiler,
    )
    diff_model = instantiate(
        hparams.model,
        _recursive_=False,
    )
    trainer.fit(diff_model, dm)


if __name__ == "__main__":
    main()
