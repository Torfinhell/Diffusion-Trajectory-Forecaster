"""Webdataset loading pipeline, kept free of JAX/waymax imports.

Grain's `mp_prefetch` reconstructs the source dataset in spawned worker
processes via `cloudpickle`, which requires importing this module fresh in
each worker. If this module (transitively) imported `jax`/`waymax`, every
worker would initialize its own JAX/XLA/CUDA backend, contending with the
already-running backend in the main training process and deadlocking
(observed as workers stuck at GPU init holding ~60 threads each). Keeping
this module's import graph limited to `webdataset`/`grain`/`numpy` avoids
that entirely.
"""

import dataclasses
import logging
import random

import grain.python as grain
import webdataset as wds
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset as grain_dataset

from src.data_module.storage import decode_sample

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
    def __init__(self, open_webdataset_fn):
        super().__init__()
        self._it = iter(open_webdataset_fn())
        self._ctx = base.IteratorContext()

    def __next__(self):
        return next(self._it)

    def get_state(self):
        return {}

    def set_state(self, _state):
        pass


@dataclasses.dataclass(frozen=True)
class _WdsPipelineSpec:
    """Plain, picklable recipe for rebuilding a webdataset pipeline.

    Carrying only primitive fields (rather than a live `WaymoWebDataset`,
    which references S3 clients, Hydra config nodes, etc.) keeps this class
    cheaply `cloudpickle`-able so it survives Grain's spawn-based
    `mp_prefetch` workers without dragging Hydra/S3 state across the process
    boundary. The actual `wds.WebDataset` pipeline is rebuilt fresh inside
    the worker, mirroring how `wds.split_by_worker` expects to run.
    """

    part: str
    sources: list

    def open(self):
        ds = wds.WebDataset(
            self.sources,
            shardshuffle=len(self.sources) if self.part == "train" else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        ).decode()
        if self.part == "train":
            ds = ds.compose(shuffle_entire_shard_once)
        return ds.map(decode_sample)


class WebDatasetGrainSource(grain.IterDataset):
    def __init__(self, dataset: "WaymoWebDataset"):
        super().__init__()
        meta = dataset.load_meta()
        self._spec = _WdsPipelineSpec(
            part=dataset.part,
            sources=dataset._shard_sources(meta),
        )

    def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
        """Assigns a per-worker subset of shards.

        Grain's `mp_prefetch` calls this with `slice(worker_index, None,
        num_workers)` so each worker handles a disjoint subset of shards
        (mirroring `wds.split_by_worker`), instead of every worker reading
        every shard.
        """
        del sequential_slice
        sources = self._spec.sources[sl]
        self._spec = dataclasses.replace(self._spec, sources=sources)

    def __iter__(self):
        return _WdsIterator(self._spec.open)
