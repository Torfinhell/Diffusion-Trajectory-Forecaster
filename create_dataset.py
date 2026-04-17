import logging
import pickle
import shutil
import subprocess
from pathlib import Path
import json

import hydra
from hydra.utils import get_original_cwd, to_absolute_path

from src.data_module.dataset_creation import (
    CHUNKED_DATASET_FORMAT,
    SINGLE_FILE_CHUNKED_DATASET_FORMAT,
    SINGLE_FILE_INDEX_MAGIC,
    SINGLE_FILE_INDEX_TRAILER,
    WEBDATASET_FORMAT,
    WEBDATASET_INDEX_FILENAME,
    iter_processed_samples,
    save_processed_samples,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_FLUSH_EVERY = 512


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


def _validate_shared_dataset_layout(processed_paths, dvc_files):
    dataset_dirs = {path.parent for path in processed_paths}
    if len(dataset_dirs) != 1:
        raise RuntimeError(
            "Expected all processed split outputs to live in the same directory, got: "
            + ", ".join(str(path) for path in sorted(dataset_dirs, key=str))
        )

    if len(dvc_files) != 1:
        raise RuntimeError(
            "Expected all splits to reference the same DVC file, got: "
            + ", ".join(str(path) for path in sorted(dvc_files, key=str))
        )

    dataset_dir = next(iter(dataset_dirs))
    dvc_file = next(iter(dvc_files))
    expected_dvc_file = dataset_dir.with_name(dataset_dir.name + ".dvc")
    if dvc_file != expected_dvc_file:
        raise RuntimeError(
            f"Configured dvc file {dvc_file} does not match the directory artifact "
            f"created by `dvc add {dataset_dir}` ({expected_dvc_file})."
        )

    return dataset_dir, dvc_file


def _read_num_samples(processed_path: Path) -> int:
    if processed_path.is_dir():
        index_path = processed_path / WEBDATASET_INDEX_FILENAME
        if not index_path.exists():
            raise RuntimeError(f"WebDataset index not found at {index_path}")
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if loaded.get("format") == WEBDATASET_FORMAT:
            return int(loaded.get("num_samples", 0))
        raise RuntimeError(f"Unsupported dataset format at {processed_path}")

    with processed_path.open("rb") as file:
        file.seek(0, 2)
        file_size = file.tell()
        trailer_size = SINGLE_FILE_INDEX_TRAILER.size
        if file_size >= trailer_size:
            file.seek(file_size - trailer_size)
            index_offset, magic = SINGLE_FILE_INDEX_TRAILER.unpack(file.read(trailer_size))
            if magic == SINGLE_FILE_INDEX_MAGIC:
                file.seek(index_offset)
                loaded = pickle.load(file)
                if loaded.get("format") == SINGLE_FILE_CHUNKED_DATASET_FORMAT:
                    return int(loaded.get("num_samples", 0))

    with processed_path.open("rb") as file:
        loaded = pickle.load(file)

    if isinstance(loaded, list):
        return len(loaded)
    if isinstance(loaded, dict) and loaded.get("format") == CHUNKED_DATASET_FORMAT:
        return int(loaded.get("num_samples", 0))
    raise RuntimeError(f"Unsupported dataset format at {processed_path}")


@hydra.main(version_base=None, config_path="src/configs", config_name="create_dataset")
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Dataset creation config: %s", cfg)

    repo_root = _repo_root()
    _require_cli("dvc")
    _require_cli("git")
    processed_artifacts = []
    dvc_files = set()
    flush_every = int(getattr(cfg, "flush_every", DEFAULT_FLUSH_EVERY))
    storage_format = str(getattr(cfg, "storage_format", "directory_chunks"))

    for split in ("train", "val", "test"):
        artifact_cfg = cfg.data[split]
        creation_cfg = cfg.dataset_creation[split]
        processed_path = _resolve_repo_path(artifact_cfg.processed_path)
        dvc_file = _resolve_repo_path(artifact_cfg.dvc_file)
        dvc_files.add(dvc_file)

        LOGGER.info("Creating %s split from raw data", split)
        samples = iter_processed_samples(
            raw_data_url=creation_cfg.raw_data_url,
            waymax_conf_version=creation_cfg.waymax_conf_version,
            num_states=creation_cfg.num_states,
            max_num_objects=creation_cfg.max_num_objects,
            extract_scene=creation_cfg.extract_scene,
            preprocess_kwargs=_to_plain_data(creation_cfg.preprocessing),
        )

        LOGGER.info(
            "Streaming %s split samples to %s with flush_every=%s storage_format=%s",
            split,
            processed_path,
            flush_every,
            storage_format,
        )
        created_artifact = save_processed_samples(
            processed_path,
            samples,
            flush_every=flush_every,
            storage_format=storage_format,
        )
        created_artifact = Path(created_artifact)
        processed_artifacts.append(created_artifact)
        num_samples = _read_num_samples(created_artifact)
        if num_samples <= 0:
            raise RuntimeError(
                f"Created empty {split} dataset at {created_artifact}. "
                "Dataset creation stopped before producing any samples."
            )

    dataset_dir, dvc_file = _validate_shared_dataset_layout(processed_artifacts, dvc_files)

    LOGGER.info("Tracking processed dataset directory with DVC: %s", dataset_dir)
    _run_logged(["dvc", "add", str(dataset_dir.relative_to(repo_root))], cwd=repo_root)

    gitignore_path = dataset_dir.parent / ".gitignore"
    LOGGER.info("Adding shared DVC metadata to git")
    _run_logged(
        [
            "git",
            "add",
            str(dvc_file.relative_to(repo_root)),
            str(gitignore_path.relative_to(repo_root)),
        ],
        cwd=repo_root,
    )

    LOGGER.info("Pushing tracked datasets to the default DVC remote")
    _run_logged(["dvc", "push"], cwd=repo_root)
    LOGGER.info("Dataset creation completed successfully")


if __name__ == "__main__":
    main()
