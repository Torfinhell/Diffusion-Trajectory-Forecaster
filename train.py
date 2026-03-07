import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.models import DiffusionModel
from src.utils import process_hparams


@hydra.main(version_base=None, config_name="ddpm_baseline", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=True)
    logger = instantiate(hparams.logger)
    dm = DiffusionTrackerDataModule(hparams.datasets, hparams.dataloaders)
    dm.setup("fit")
    trainer = Trainer(
        accelerator="auto",
        max_epochs=hparams.trainer.num_epochs,
        logger=logger,
        callbacks=[],
        log_every_n_steps=hparams.trainer.log_every_n_steps,
        enable_progress_bar=True,
        limit_train_batches=hparams.trainer.train_epoch_len,
        limit_val_batches=hparams.trainer.val_epoch_len,
        reload_dataloaders_every_n_epochs=cfg.trainer.generate_every_epoch,
    )
    diff_model = instantiate(
    hparams.model,
    cfg_metrics=hparams.metrics,
    grad_clip=hparams.trainer.gradient_clip_val,
    vis_cfg=hparams.visual,
    cfg_model=hparams.architectures,
     _recursive_=False,
)
    trainer.fit(diff_model, dm)


if __name__ == "__main__":
    main()
