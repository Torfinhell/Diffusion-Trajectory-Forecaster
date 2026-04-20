import hashlib
import os
import socket
import subprocess
import sys
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


def _run_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _git_metadata() -> dict:
    commit = _run_command(["git", "rev-parse", "HEAD"])
    short_commit = _run_command(["git", "rev-parse", "--short", "HEAD"])
    branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    remote_url = _run_command(["git", "remote", "get-url", "origin"])
    status = _run_command(["git", "status", "--short"])
    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "remote_url": remote_url,
        "is_dirty": bool(status),
        "status_short": status or "",
    }


def _runtime_metadata() -> dict:
    hydra_runtime = HydraConfig.get().runtime
    return {
        "cwd": os.getcwd(),
        "original_cwd": hydra_runtime.cwd,
        "output_dir": hydra_runtime.output_dir,
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
    }


def _write_git_snapshot(metadata: dict) -> Path:
    git_info = metadata.get("git", {})
    snapshot_path = Path("git_snapshot.txt")
    lines = [
        f"commit: {git_info.get('commit')}",
        f"short_commit: {git_info.get('short_commit')}",
        f"branch: {git_info.get('branch')}",
        f"remote_url: {git_info.get('remote_url')}",
        f"is_dirty: {git_info.get('is_dirty')}",
        "",
        "status_short:",
        git_info.get("status_short", ""),
        "",
    ]
    snapshot_path.write_text("\n".join(lines), encoding="utf-8")
    return snapshot_path


def log_run_metadata(logger, cfg) -> None:
    if logger is None:
        return

    runtime_choices = dict(HydraConfig.get().runtime.choices)
    dataset_name = runtime_choices.get("data")
    metadata = {
        "hydra_choices": runtime_choices,
        "git": _git_metadata(),
        "runtime": _runtime_metadata(),
        "dataset": {
            "name": dataset_name,
            "splits": _dataset_metadata(cfg),
        },
    }

    resolved_config_path = Path("resolved_config.yaml")
    resolved_config_path.write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    if hasattr(logger, "upload_artifact"):
        logger.upload_artifact(
            name="resolved_config",
            path=resolved_config_path,
            metadata=metadata,
        )
        logger.upload_artifact(
            name="git_snapshot",
            path=_write_git_snapshot(metadata),
            metadata=metadata.get("git", {}),
        )
