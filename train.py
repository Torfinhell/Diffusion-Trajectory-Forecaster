import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.models import DiffusionModel
from src.utils import process_hparams


@hydra.main(version_base=None, config_name="ddpm", config_path="src/configs")
def main(cfg: DictConfig) -> None:
    hparams = process_hparams(cfg, print_hparams=True)
    logger = WandbLogger(
        project=hparams.project, name=hparams.experiment, mode=hparams.logging
    )
    dm = DiffusionTrackerDataModule(hparams.data_module)
    dm.setup("fit")
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=hparams.trainer.num_epochs,
        logger=logger,
        callbacks=[],
        log_every_n_steps=hparams.trainer.log_every_n_steps,
        gradient_clip_val=hparams.trainer.gradient_clip_val,
        enable_progress_bar=True,
        limit_train_batches=hparams.trainer.train_epoch_len,
        limit_val_batches=hparams.trainer.val_epoch_len,
        reload_dataloaders_every_n_epochs=cfg.datamodule_cfg.generate_every_epoch,
    )
    diff_model = DiffusionModel(hparams.model, seed=hparams.trainer.seed)
    trainer.fit(diff_model, dm)
