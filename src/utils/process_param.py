import hashlib
from copy import deepcopy
from pathlib import Path

import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def process_hparams(hparams, print_hparams=True):
    OmegaConf.set_struct(hparams, False)
    if hparams.trainer.logging == "online":
        hparams.show = False

    """Create HParam ID for saving and loading checkpoints"""
    hashable_config = deepcopy(hparams)
    id = hashlib.sha1(
        repr(sorted(hashable_config.__dict__.items())).encode()
    ).hexdigest()
    hparams.hparams_id = id
    OmegaConf.set_struct(hparams, True)
    if print_hparams:
        print(OmegaConf.to_container(hparams, resolve=True))
    return hparams


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


def log_run_metadata(logger, cfg) -> None:
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
