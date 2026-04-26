import dataclasses
import json
import logging
import shutil
from pathlib import Path

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
    def _read_num_samples(output_root: Path) -> int:
        index_path = output_root / WEBDATASET_INDEX_FILENAME
        assert index_path.exists(), f"WebDataset index not found at {index_path}"
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        assert  loaded.get("format") == WEBDATASET_FORMAT, f"Unsupported dataset format in {index_path}"
        return int(loaded.get("num_samples", 0))

    @classmethod
    def ensure_local_webdataset(
        cls,
        split: str,
        split_cfg,
        processed_path: str,
        creation_cfg=None,
        flush_every: int = DEFAULT_FLUSH_EVERY,
    ) -> Path:
        output_root = Path(to_absolute_path(processed_path))
        index_path = output_root / WEBDATASET_INDEX_FILENAME
        if index_path.exists():
            return output_root

        if creation_cfg is not None:
            LOGGER.info(
                "WebDataset artifact not found at %s. Creating %s split locally.",
                output_root,
                split,
            )
            builder = cls(flush_every=int(split_cfg.get("flush_every", flush_every)))
            builder.create_split(
                split=split,
                artifact_cfg=split_cfg,
                creation_cfg=creation_cfg,
            )
            if not index_path.exists():
                raise FileNotFoundError(f"WebDataset index not found at {index_path}.")
            return output_root

        raise FileNotFoundError(
            f"WebDataset artifact not found at {output_root} and no creation config was provided."
        )

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
        start_index,
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
        start_index = int(start_index)
        num_states = int(num_states)
        produced = 0
        total_to_read = start_index + num_states

        for index in tqdm(range(total_to_read), total=total_to_read, desc="Creating dataset"):
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
                    total_to_read,
                    exc,
                )
                break

            if index < start_index:
                continue

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
        assert self.flush_every > 0, "flush_every must be a positive integer"
        output_root = Path(processed_path)
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
            start_index=getattr(creation_cfg, "start_index", 0),
            num_states=creation_cfg.num_states,
            max_num_objects=creation_cfg.max_num_objects,
            extract_scene=creation_cfg.extract_scene,
            preprocess_kwargs=creation_cfg.preprocessing,
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
                creation_cfg=cfg.data[split].creation,
            )
        return created

    @staticmethod
    def _unwrap_payload(sample: dict):
        payload = sample["pth"]
        if isinstance(payload, dict) and "__key__" in payload:
            payload = {key: value for key, value in payload.items() if key != "__key__"}
        return payload

    @classmethod
    def build_webdataset(cls, split: str, split_cfg, is_train: bool):
        output_root = cls.ensure_local_webdataset(
            split=split,
            split_cfg=split_cfg,
            processed_path=split_cfg.processed_path,
            creation_cfg=split_cfg.get("creation"),
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
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        ).decode()

        if is_train:
            dataset = dataset.shuffle(int(split_cfg.get("shuffle_buffer", 1000)))

        dataset = dataset.map(cls._unwrap_payload)
        return SizedIterableDataset(dataset, int(metadata.get("num_samples", 0)))
