import dataclasses
import io
import logging
import math
import pickle
import random
from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import webdataset as wds
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import IterableDataset, get_worker_info
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


class WaymoWebDataset(IterableDataset):
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
        if self.meta is not None:
            return self.meta
        if self.remote is not None:
            assert self.data_access in ("stream", "cache"), self.data_access
            assert (
                self.remote.exists()
            ), f"{self.remote.prefix} missing; run create_dataset"
            if self.data_access == "cache":
                self.meta = self.remote.sync_to(self.local)
                return self.meta
            if (self.local / "index.json").is_file():
                self.meta = read_local_index(self.local)
                return self.meta
            self.meta = self.remote.read_index()
            return self.meta
        assert self.local.exists(), f"{self.local} missing; run create_dataset"
        self.meta = read_local_index(self.local)
        return self.meta

    def _shard_sources(self, meta: dict) -> list[str]:
        if self.data_access == "stream":
            assert self.remote is not None
            return self.remote.stream_sources(meta)

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
        )

        def custom_decoder(key, data):
            if key.endswith(".pickle"):
                return pickle.loads(data)
            if key.endswith(".npy"):
                return np.load(io.BytesIO(data))
            return None

        ds = ds.decode(custom_decoder)

        if self.part == "train":
            ds = ds.compose(shuffle_entire_shard_once)

        if self.part == "train":
            ds = ds.compose(shuffle_entire_shard_once)

        return ds.map(decode_sample)

    def __iter__(self):
        worker_ds = self._open_webdataset()
        return iter(worker_ds)

    def __len__(self):
        meta = self.load_meta()
        worker_info = get_worker_info()
        if worker_info is None:
            return int(meta["num_samples"])
        return int(math.ceil(int(meta["num_samples"]) / float(worker_info.num_workers)))
