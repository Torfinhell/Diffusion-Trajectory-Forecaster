import json
import subprocess
from pathlib import Path

import webdataset as wds
from hydra.utils import to_absolute_path
from torch.utils.data import IterableDataset

from src.data_module.dataset_creation import (
    WEBDATASET_FORMAT,
    WEBDATASET_INDEX_FILENAME,
    resolve_webdataset_output_root,
)


class SizedIterableDataset(IterableDataset):
    def __init__(self, dataset, length: int):
        super().__init__()
        self.dataset = dataset
        self.length = int(length)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length


def _ensure_local_webdataset(processed_path: str, dvc_file: str | None = None) -> Path:
    output_root = resolve_webdataset_output_root(Path(to_absolute_path(processed_path)))
    index_path = output_root / WEBDATASET_INDEX_FILENAME
    if index_path.exists():
        return output_root

    if dvc_file is None:
        raise FileNotFoundError(
            f"WebDataset artifact not found at {output_root} and no dvc_file was provided."
        )

    print(f"WebDataset artifact not found at {output_root}. Pulling with DVC.")
    try:
        subprocess.run(["dvc", "pull", dvc_file], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not run `dvc pull` because the `dvc` CLI is not installed."
        ) from exc

    if not index_path.exists():
        raise FileNotFoundError(f"WebDataset index not found at {index_path}.")
    return output_root


def _unwrap_payload(sample: dict):
    payload = sample["pth"]
    if isinstance(payload, dict) and "__key__" in payload:
        payload = {key: value for key, value in payload.items() if key != "__key__"}
    return payload


def build_webdataset(split_cfg, is_train: bool):
    output_root = _ensure_local_webdataset(
        processed_path=split_cfg.processed_path,
        dvc_file=split_cfg.get("dvc_file"),
    )
    index_path = output_root / WEBDATASET_INDEX_FILENAME
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    if metadata.get("format") != WEBDATASET_FORMAT:
        raise RuntimeError(f"Unsupported WebDataset format in {index_path}.")

    shard_paths = sorted(output_root.glob(metadata.get("shard_glob", "shard-*.tar")))
    if not shard_paths:
        raise FileNotFoundError(f"No WebDataset shards found under {output_root}.")

    dataset = wds.WebDataset(
        [str(path) for path in shard_paths],
        shardshuffle=len(shard_paths) if is_train else False,
    ).decode()

    if is_train:
        dataset = dataset.shuffle(int(split_cfg.get("shuffle_buffer", 1000)))

    dataset = dataset.map(_unwrap_payload)
    return SizedIterableDataset(dataset, int(metadata.get("num_samples", 0)))
