import dataclasses
import logging
import random
from itertools import islice
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import webdataset as wds
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset as grain_dataset
from hydra.utils import instantiate, to_absolute_path
from tqdm.auto import tqdm
from waymax import config, dataloader

from src.data_module.data_process import data_process_scenarios_batch
from src.data_module.storage import (
    S3Storage,
    decode_sample,
    read_local_index,
    write_webdataset,
)

LOGGER = logging.getLogger(__name__)


def shuffle_entire_shard_once(src):
    """Shuffle samples within each tar shard; shard order comes from shardshuffle."""
    current_shard = None
    shard_samples = []

    def flush():
        random.shuffle(shard_samples)
        yield from shard_samples
        shard_samples.clear()

    for sample in src:
        sample_shard = sample.get("__url__")
        if current_shard is not None and sample_shard != current_shard:
            yield from flush()
        current_shard = sample_shard
        shard_samples.append(sample)
    if shard_samples:
        yield from flush()


class _WdsIterator(grain_dataset.DatasetIterator):
    def __init__(self, wds_iter):
        super().__init__()
        self._it = iter(wds_iter)
        self._ctx = base.IteratorContext()

    def __next__(self):
        return next(self._it)

    def get_state(self):
        return {}

    def set_state(self, _state):
        pass


class WebDatasetGrainSource(grain.IterDataset):
    def __init__(self, dataset: "WaymoWebDataset"):
        super().__init__()
        self._dataset = dataset

    def __iter__(self):
        return _WdsIterator(self._dataset._open_webdataset())


class WaymoWebDataset:
    NAME = "waymo_webdataset"

    def __init__(
        self,
        part: str,
        path: str,
        flush_every: int = 512,
        creation_cfg=None,
        allow_upload: bool = False,
        s3_url: str | None = None,
        data_access: str | None = "local",
    ):
        assert part in ("train", "val", "test")
        self.part = part
        self.flush_every = int(flush_every)
        self.creation_cfg = creation_cfg
        self.allow_upload = bool(allow_upload)
        self.data_access = data_access or "local"
        self.s3_root = s3_url
        self.local = Path(to_absolute_path(path))
        self.remote = S3Storage.for_split(self.s3_root, part) if self.s3_root else None
        self.meta = None
        self.ensure_artifact()

    def ensure_artifact(self):
        if self.local.exists() or (self.remote and self.remote.exists()):
            LOGGER.info("Skipping %s build: artifact exists", self.part)
            return
        assert (
            self.creation_cfg is not None
        ), f"{self.part}: creation_cfg required to build"
        write_webdataset(
            self.local,
            self._iter_samples(),
            self.flush_every,
            self.remote if self.allow_upload else None,
        )

    def _iter_samples(self):
        c = self.creation_cfg
        preprocess = instantiate(c.preprocessing)
        it = dataloader.simulator_state_generator(
            config=dataclasses.replace(
                getattr(config, c.waymax_conf_version),
                path=str(c.raw_data_url),
                max_num_objects=c.max_num_objects,
            )
        )
        for _ in range(int(getattr(c, "start_index", 0))):
            next(it)
        batch_size = int(getattr(c, "batch_size", 1))
        pending = []
        for state in tqdm(
            islice(it, int(c.num_states)),
            total=int(c.num_states),
            desc=f"Creating {self.part}",
        ):
            pending.append(state)
            if len(pending) < batch_size:
                continue
            processed = data_process_scenarios_batch(
                jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *pending),
                **preprocess,
            )
            for i in range(processed["agent_past"].shape[0]):
                row = jax.tree_util.tree_map(lambda x, j=i: x[j], processed)
                if c.extract_scene or i < 5:
                    row = {"scenario": pending[i], **row}
                yield row
            pending = []
        if pending:
            processed = data_process_scenarios_batch(
                jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *pending),
                **preprocess,
            )
            for i in range(processed["agent_past"].shape[0]):
                row = jax.tree_util.tree_map(lambda x, j=i: x[j], processed)
                if c.extract_scene or i < 5:
                    row = {"scenario": pending[i], **row}
                yield row

    def load_meta(self) -> dict:
        if self.remote is not None:
            assert self.data_access in ("stream", "cache"), self.data_access
            assert (
                self.remote.exists()
            ), f"{self.remote.prefix} missing; run create_dataset"
            if self.data_access == "cache":
                return self.remote.sync_to(self.local)
            if (self.local / "index.json").is_file():
                return read_local_index(self.local)
            return self.remote.read_index()
        assert self.local.exists(), f"{self.local} missing; run create_dataset"
        return read_local_index(self.local)

    def _shard_sources(self, meta: dict) -> list[str]:
        if self.data_access == "stream":
            assert self.remote is not None
            return 1
        paths = sorted(self.local.glob(meta.get("shard_glob", "shard-*.tar")))
        assert paths, self.local
        return [str(p) for p in paths]

    def _open_webdataset(self):
        meta = self.load_meta()
        sources = self._shard_sources(meta)
        ds = wds.WebDataset(
            sources,
            shardshuffle=len(sources) if self.part == "train" else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        ).decode()
        if self.part == "train":
            ds = ds.compose(shuffle_entire_shard_once)
        self.meta = meta
        return ds.map(decode_sample)

    def __len__(self):
        if self.meta is None:
            self.meta = self.load_meta()
        return int(self.meta["num_samples"])
