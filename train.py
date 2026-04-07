from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.utils import process_hparams


def _read_dvc_metadata(dvc_file: str) -> dict:
    dvc_path = Path(to_absolute_path(dvc_file))
    metadata = {
        "path": dvc_file,
        "absolute_path": str(dvc_path),
        "exists": dvc_path.exists(),
    }
    if not dvc_path.exists():
        return metadata

    with dvc_path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file) or {}
    metadata["contents"] = parsed
    return metadata


def _dataset_metadata(cfg) -> dict:
    metadata = {}
    for split, split_cfg in cfg.data.items():
        split_info = {
            "processed_path": split_cfg.processed_path,
            "processed_path_abs": to_absolute_path(split_cfg.processed_path),
            "dvc_file": split_cfg.dvc_file,
            "dvc": _read_dvc_metadata(split_cfg.dvc_file),
        }
        metadata[split] = split_info
    return metadata


def _log_run_metadata(logger, cfg) -> None:
    if logger is None:
        return

    runtime_choices = dict(HydraConfig.get().runtime.choices)
    dataset_name = runtime_choices.get("data")
    metadata = {
        "hydra_choices": runtime_choices,
        "dataset": {
            "name": dataset_name,
            "splits": _dataset_metadata(cfg),
        },
    }
    if hasattr(logger, "log_run_metadata"):
        logger.log_run_metadata(metadata)

    resolved_config_path = Path("resolved_config.yaml")
    resolved_config_path.write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    if hasattr(logger, "upload_artifact"):
        logger.upload_artifact(
            name=f"{getattr(logger, 'name', 'run')}-config",
            path=resolved_config_path,
            metadata=metadata,
        )


@hydra.main(version_base=None, config_name="ddpm_baseline", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=True)
    logger = instantiate(hparams.logger)
    _log_run_metadata(logger, hparams)
    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    progress_bar = RichProgressBar(leave=True)
    trainer = Trainer(
        accelerator="auto",
        max_epochs=hparams.trainer.num_epochs,
        logger=logger,
        callbacks=[progress_bar],
        log_every_n_steps=hparams.trainer.log_every_n_steps,
        enable_progress_bar=True,
        limit_train_batches=hparams.trainer.train_epoch_len,
        limit_val_batches=hparams.trainer.val_epoch_len,
        check_val_every_n_epoch=hparams.trainer.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=cfg.trainer.generate_every_epoch,
    )
    diff_model = instantiate(
        hparams.model,
        _recursive_=False,
    )
    trainer.fit(diff_model, dm)


if __name__ == "__main__":
    main()
