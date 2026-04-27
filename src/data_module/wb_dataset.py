import dataclasses
import json
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

import jax
import jax.numpy as jnp
import numpy as np
import webdataset as wds
from hydra.utils import to_absolute_path
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm
from waymax import config, dataloader

from src.data_module.data_process import data_process_scenarios

LOGGER = logging.getLogger(__name__)
WEBDATASET_FORMAT = "diffusion_tracker_webdataset_v2"
WEBDATASET_INDEX_FILENAME = "index.json"
DEFAULT_FLUSH_EVERY = 512
DEFAULT_REMOTE_STREAM_COMMAND = "aws s3 cp {url} -"


def _is_s3_url(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).startswith("s3://")


def _split_s3_url(url: str) -> tuple[str, str]:
    parsed = urlparse(str(url).rstrip("/"))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Unsupported S3 url: {url}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _s3_join(root_url: str, name: str) -> str:
    return f"{str(root_url).rstrip('/')}/{str(name).lstrip('/')}"


def _s3_client():
    import boto3

    return boto3.client("s3")


def _s3_upload_file(root_url: str, local_path: Path, name: str) -> None:
    bucket, prefix = _split_s3_url(root_url)
    key = f"{prefix}/{name}" if prefix else name
    _s3_client().upload_file(str(local_path), bucket, key)


def _s3_write_text(root_url: str, name: str, text: str) -> None:
    bucket, prefix = _split_s3_url(root_url)
    key = f"{prefix}/{name}" if prefix else name
    _s3_client().put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def _s3_read_text(root_url: str, name: str) -> str:
    bucket, prefix = _split_s3_url(root_url)
    key = f"{prefix}/{name}" if prefix else name
    body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"]
    return body.read().decode("utf-8")


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

    @classmethod
    def _resolve_local_output_root(cls, split_cfg) -> Path:
        processed_path = str(split_cfg.processed_path)
        if not _is_s3_url(processed_path):
            return Path(to_absolute_path(processed_path))

        local_cache_path = split_cfg.get("local_cache_path")
        if local_cache_path:
            return Path(to_absolute_path(str(local_cache_path)))

        parsed = urlparse(processed_path)
        artifact_suffix = parsed.path.lstrip("/")
        return Path("/tmp/remote_cache") / parsed.netloc / artifact_suffix

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
        assert loaded.get("format") == WEBDATASET_FORMAT, f"Unsupported dataset format in {index_path}"
        return int(loaded.get("num_samples", 0))

    @staticmethod
    def _read_num_samples_from_metadata(metadata: dict) -> int:
        assert metadata.get("format") == WEBDATASET_FORMAT, "Unsupported WebDataset format"
        return int(metadata.get("num_samples", 0))

    @staticmethod
    def _encode_sample_fields(index: int, sample: dict) -> dict:
        if "scenario" in sample:
            raise ValueError(
                "Per-field WebDataset writing does not support raw 'scenario' payloads. "
                "Use datasets with extract_scene=false."
            )

        encoded = {"__key__": f"{index:09d}"}
        for key, value in sample.items():
            encoded[f"{key}.npy"] = np.asarray(value)
        return encoded

    @classmethod
    def ensure_local_webdataset(
        cls,
        split: str,
        split_cfg,
        creation_cfg=None,
        flush_every: int = DEFAULT_FLUSH_EVERY,
    ) -> Path:
        output_root = cls._resolve_local_output_root(split_cfg)
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

    def save_processed_samples(
        self,
        processed_path,
        samples,
        s3_path: str | None = None,
    ):
        assert self.flush_every > 0, "flush_every must be a positive integer"
        output_root = Path(processed_path)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        shard_pattern = str(output_root / "shard-%06d.tar")
        total_samples = 0
        num_shards = 0

        def upload_shard(shard_path_str: str) -> None:
            nonlocal num_shards
            num_shards += 1
            if s3_path is None:
                return

            shard_path = Path(shard_path_str)
            LOGGER.info(
                "Uploading shard %s to %s",
                shard_path.name,
                _s3_join(s3_path, shard_path.name),
            )
            _s3_upload_file(s3_path, shard_path, shard_path.name)
            shard_path.unlink(missing_ok=True)

        with wds.ShardWriter(
            shard_pattern,
            maxcount=self.flush_every,
            post=upload_shard,
            verbose=0,
        ) as sink:
            for index, sample in enumerate(samples):
                sink.write(self._encode_sample_fields(index, sample))
                total_samples += 1

        index_path = output_root / WEBDATASET_INDEX_FILENAME
        metadata = {
            "format": WEBDATASET_FORMAT,
            "num_samples": total_samples,
            "num_shards": num_shards,
            "shard_glob": "shard-*.tar",
            "shard_pattern": "shard-%06d.tar",
        }
        metadata_text = json.dumps(metadata, indent=2, sort_keys=True)
        index_path.write_text(metadata_text, encoding="utf-8")
        if s3_path is not None:
            _s3_write_text(s3_path, WEBDATASET_INDEX_FILENAME, metadata_text)
            shutil.rmtree(output_root, ignore_errors=True)
        return output_root

    def create_split(self, split: str, artifact_cfg, creation_cfg) -> Path:
        processed_path = self._resolve_local_output_root(artifact_cfg)
        s3_path = str(artifact_cfg.processed_path) if _is_s3_url(artifact_cfg.processed_path) else None

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
        output_root = Path(
            self.save_processed_samples(
                processed_path=processed_path,
                samples=samples,
                s3_path=s3_path,
            )
        )
        if s3_path is not None:
            LOGGER.info(
                "Uploaded %s split to %s",
                split,
                s3_path,
            )
        if s3_path is not None:
            metadata = json.loads(_s3_read_text(s3_path, WEBDATASET_INDEX_FILENAME))
            num_samples = self._read_num_samples_from_metadata(metadata)
        else:
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
    def _decode_sample_fields(sample: dict):
        metadata_keys = {"__key__", "__url__", "__local_path__"}
        decoded = {}
        for key, value in sample.items():
            if key in metadata_keys:
                continue
            if not key.endswith(".npy"):
                raise ValueError(f"Unsupported WebDataset field '{key}'")
            decoded[key[:-4]] = value
        return decoded

    @classmethod
    def _build_remote_shard_sources(cls, s3_path: str, metadata: dict):
        shard_pattern = str(metadata.get("shard_pattern", "shard-%06d.tar"))
        num_shards = int(metadata.get("num_shards", 0))
        shard_sources = []
        for shard_idx in range(num_shards):
            shard_name = shard_pattern % shard_idx
            shard_sources.append(
                "pipe:" + DEFAULT_REMOTE_STREAM_COMMAND.format(
                    url=_s3_join(s3_path, shard_name),
                )
            )
        return shard_sources

    @classmethod
    def _read_remote_metadata(cls, split: str, split_cfg):
        s3_path = str(split_cfg.processed_path)
        try:
            metadata = json.loads(_s3_read_text(s3_path, WEBDATASET_INDEX_FILENAME))
        except Exception:
            creation_cfg = split_cfg.get("creation")
            if creation_cfg is None:
                raise FileNotFoundError(
                    f"Remote WebDataset index not found at "
                    f"{_s3_join(s3_path, WEBDATASET_INDEX_FILENAME)}."
                )
            LOGGER.info(
                "Remote WebDataset not found at %s. Creating %s split.",
                s3_path,
                split,
            )
            builder = cls(flush_every=int(split_cfg.get("flush_every", DEFAULT_FLUSH_EVERY)))
            builder.create_split(
                split=split,
                artifact_cfg=split_cfg,
                creation_cfg=creation_cfg,
            )
            metadata = json.loads(_s3_read_text(s3_path, WEBDATASET_INDEX_FILENAME))

        if metadata.get("format") != WEBDATASET_FORMAT:
            raise RuntimeError(
                f"Unsupported WebDataset format in "
                f"{_s3_join(s3_path, WEBDATASET_INDEX_FILENAME)}."
            )
        return metadata

    @classmethod
    def build_webdataset(cls, split: str, split_cfg, is_train: bool):
        if _is_s3_url(split_cfg.processed_path):
            s3_path = str(split_cfg.processed_path)
            metadata = cls._read_remote_metadata(split, split_cfg)
            shard_sources = cls._build_remote_shard_sources(s3_path, metadata)
            if not shard_sources:
                raise FileNotFoundError(
                    f"No WebDataset shards found under {s3_path}."
                )
        else:
            output_root = cls.ensure_local_webdataset(
                split=split,
                split_cfg=split_cfg,
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
                raise FileNotFoundError(
                    f"No WebDataset shards found under {output_root}."
                )
            shard_sources = [str(path) for path in shard_paths]

        dataset = wds.WebDataset(
            shard_sources,
            shardshuffle=len(shard_sources) if is_train else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        ).decode()

        if is_train:
            dataset = dataset.shuffle(int(split_cfg.get("shuffle_buffer", 1000)))

        dataset = dataset.map(cls._decode_sample_fields)
        return SizedIterableDataset(dataset, int(metadata.get("num_samples", 0)))
