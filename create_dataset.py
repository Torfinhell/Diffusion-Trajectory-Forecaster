import logging
import shutil
import subprocess
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path

from src.data_module.dataset_creation import (
    create_processed_samples,
    save_processed_samples,
)


LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(get_original_cwd())


def _resolve_repo_path(path_value: str) -> Path:
    return Path(to_absolute_path(path_value))


def _run_logged(cmd, cwd):
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _to_plain_data(node):
    if hasattr(node, "items"):
        return {key: _to_plain_data(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_to_plain_data(value) for value in node]
    return node


def _require_cli(name):
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required CLI '{name}' is not available in PATH. "
            "Install it before running dataset creation."
        )


@hydra.main(version_base=None, config_path="src/configs", config_name="create_dataset")
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Dataset creation config: %s", cfg)

    repo_root = _repo_root()
    _require_cli("dvc")
    _require_cli("git")
    for split in ("train", "val", "test"):
        artifact_cfg = cfg.data[split]
        creation_cfg = cfg.dataset_creation[split]
        processed_path = _resolve_repo_path(artifact_cfg.processed_path)
        dvc_file = _resolve_repo_path(artifact_cfg.dvc_file)

        LOGGER.info("Creating %s split from raw data", split)
        samples = create_processed_samples(
            raw_data_url=creation_cfg.raw_data_url,
            waymax_conf_version=creation_cfg.waymax_conf_version,
            num_states=creation_cfg.num_states,
            max_num_objects=creation_cfg.max_num_objects,
            extract_scene=creation_cfg.extract_scene,
            preprocess_kwargs=_to_plain_data(creation_cfg.preprocessing),
        )

        LOGGER.info(
            "Saving %s processed %s samples to %s",
            len(samples),
            split,
            processed_path,
        )
        save_processed_samples(processed_path, samples)

        LOGGER.info("Tracking %s split with DVC", split)
        _run_logged(["dvc", "add", str(processed_path)], cwd=repo_root)

        expected_dvc_file = processed_path.with_name(processed_path.name + ".dvc")
        if dvc_file != expected_dvc_file:
            LOGGER.warning(
                "Configured dvc file %s does not match the default file created by `dvc add` (%s).",
                dvc_file,
                expected_dvc_file,
            )

        LOGGER.info("Adding %s split DVC metadata to git", split)
        _run_logged(
            ["git", "add", str(dvc_file.relative_to(repo_root)), ".gitignore"],
            cwd=repo_root,
        )

    LOGGER.info("Pushing tracked datasets to the default DVC remote")
    _run_logged(["dvc", "push"], cwd=repo_root)
    LOGGER.info("Dataset creation completed successfully")


if __name__ == "__main__":
    main()
