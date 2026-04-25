import dataclasses
import json
import logging
import shutil
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import webdataset as wds
from hydra.utils import to_absolute_path
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm
from waymax import config, dataloader

from src.data_module.data_process import data_process_scenarios

LOGGER = logging.getLogger(__name__)
WEBDATASET_FORMAT = "diffusion_tracker_webdataset_v1"
WEBDATASET_INDEX_FILENAME = "index.json"
DEFAULT_FLUSH_EVERY = 512


class SizedIterableDataset(IterableDataset):
    def __init__(self, dataset, length: int):
        super().__init__()
        self.dataset = dataset
        self.length = int(length)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length


class Dataset:
    def __init__(self, flush_every: int = DEFAULT_FLUSH_EVERY):
        self.flush_every = int(flush_every)

    @staticmethod
    def _strip_leading_batch_axis(sample):
        cleaned = {}
        for key, value in sample.items():
            if key == "scenario":
                cleaned[key] = value
                continue
            arr = jnp.asarray(value)
            if arr.ndim > 0 and arr.shape[0] == 1:
                cleaned[key] = jnp.squeeze(arr, axis=0)
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _to_plain_data(node):
        if hasattr(node, "items"):
            return {key: Dataset._to_plain_data(value) for key, value in node.items()}
        if isinstance(node, list):
            return [Dataset._to_plain_data(value) for value in node]
        return node

    @staticmethod
    def resolve_webdataset_output_root(processed_path) -> Path:
        processed_path = Path(processed_path)
        suffix = processed_path.suffix or ".data"
        return processed_path.with_suffix(f"{suffix}.wds")

    @staticmethod
    def _read_num_samples(output_root: Path) -> int:
        index_path = output_root / WEBDATASET_INDEX_FILENAME
        if not index_path.exists():
            raise RuntimeError(f"WebDataset index not found at {index_path}")
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if loaded.get("format") != WEBDATASET_FORMAT:
            raise RuntimeError(f"Unsupported dataset format in {index_path}")
        return int(loaded.get("num_samples", 0))

    @staticmethod
    def build_waymax_iterator(raw_data_url, waymax_conf_version, max_num_objects):
        waymax_config = getattr(config, waymax_conf_version)
        waymax_config = dataclasses.replace(
            waymax_config,
            path=str(raw_data_url),
            max_num_objects=max_num_objects,
        )
        return dataloader.simulator_state_generator(config=waymax_config)

    def iter_processed_samples(
        self,
        raw_data_url,
        waymax_conf_version,
        num_states,
        max_num_objects,
        extract_scene,
        preprocess_kwargs,
    ):
        iterator = self.build_waymax_iterator(
            raw_data_url=raw_data_url,
            waymax_conf_version=waymax_conf_version,
            max_num_objects=max_num_objects,
        )
        produced = 0
        for index in tqdm(range(num_states), total=num_states, desc="Creating dataset"):
            try:
                state = next(iterator)
            except Exception as exc:
                if index == 0:
                    raise RuntimeError(
                        "Dataset creation failed before producing the first sample. "
                        f"raw_data_url={raw_data_url}, waymax_conf_version={waymax_conf_version}"
                    ) from exc
                LOGGER.warning(
                    "Stopped dataset creation after %s/%s states: %s",
                    index,
                    num_states,
                    exc,
                )
                break

            batched_scenario = jax.tree_util.tree_map(lambda x: x[None, ...], state)
            processed = self._strip_leading_batch_axis(
                data_process_scenarios(batched_scenario, **preprocess_kwargs)
            )
            if extract_scene:
                processed = {"scenario": state, **processed}
            yield processed
            produced += 1
            if produced >= int(num_states):
                break

    def save_processed_samples(self, processed_path, samples):
        if self.flush_every <= 0:
            raise RuntimeError("flush_every must be a positive integer.")
        output_root = self.resolve_webdataset_output_root(processed_path)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        shard_pattern = str(output_root / "shard-%06d.tar")
        total_samples = 0

        with wds.ShardWriter(shard_pattern, maxcount=self.flush_every) as sink:
            for index, sample in enumerate(samples):
                sink.write(
                    {
                        "__key__": f"{index:09d}",
                        "pth": wds.torch_dumps(sample),
                    }
                )
                total_samples += 1

        index_path = output_root / WEBDATASET_INDEX_FILENAME
        metadata = {
            "format": WEBDATASET_FORMAT,
            "num_samples": total_samples,
            "num_shards": len(list(output_root.glob("shard-*.tar"))),
            "shard_glob": "shard-*.tar",
        }
        index_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return output_root

    def create_split(self, split: str, artifact_cfg, creation_cfg) -> Path:
        processed_path = Path(to_absolute_path(artifact_cfg.processed_path))
        LOGGER.info("Creating %s split from raw data", split)
        samples = self.iter_processed_samples(
            raw_data_url=creation_cfg.raw_data_url,
            waymax_conf_version=creation_cfg.waymax_conf_version,
            num_states=creation_cfg.num_states,
            max_num_objects=creation_cfg.max_num_objects,
            extract_scene=creation_cfg.extract_scene,
            preprocess_kwargs=self._to_plain_data(creation_cfg.preprocessing),
        )

        LOGGER.info(
            "Streaming %s split samples to %s with flush_every=%s",
            split,
            processed_path,
            self.flush_every,
        )
        output_root = Path(self.save_processed_samples(processed_path, samples))
        num_samples = self._read_num_samples(output_root)
        if num_samples <= 0:
            raise RuntimeError(
                f"Created empty {split} dataset at {output_root}. "
                "Dataset creation stopped before producing any samples."
            )
        return output_root

    def create_splits(self, cfg, splits):
        created = {}
        for split in splits:
            created[split] = self.create_split(
                split=split,
                artifact_cfg=cfg.data[split],
                creation_cfg=cfg.dataset_creation[split],
            )
        return created

    @staticmethod
    def _unwrap_payload(sample: dict):
        payload = sample["pth"]
        if isinstance(payload, dict) and "__key__" in payload:
            payload = {key: value for key, value in payload.items() if key != "__key__"}
        return payload

    @classmethod
    def build_webdataset(cls, split_cfg, is_train: bool):
        output_root = cls.resolve_webdataset_output_root(
            Path(to_absolute_path(split_cfg.processed_path))
        )
        index_path = output_root / WEBDATASET_INDEX_FILENAME
        metadata = json.loads(index_path.read_text(encoding="utf-8"))
        if metadata.get("format") != WEBDATASET_FORMAT:
            raise RuntimeError(f"Unsupported WebDataset format in {index_path}.")

        shard_paths = sorted(
            output_root.glob(metadata.get("shard_glob", "shard-*.tar"))
        )
        if not shard_paths:
            raise FileNotFoundError(f"No WebDataset shards found under {output_root}.")

        dataset = wds.WebDataset(
            [str(path) for path in shard_paths],
            shardshuffle=len(shard_paths) if is_train else False,
        ).decode()

        if is_train:
            dataset = dataset.shuffle(int(split_cfg.get("shuffle_buffer", 1000)))

        dataset = dataset.map(cls._unwrap_payload)
        return SizedIterableDataset(dataset, int(metadata.get("num_samples", 0)))
