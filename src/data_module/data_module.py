import math
import os

import grain.python as grain
import numpy as np
import pytorch_lightning as L
from absl import flags
from hydra.utils import instantiate

from src.data_module.wds_loader import WebDatasetGrainSource

DATASET_SHARED_KEYS = ("allow_upload", "s3_url", "data_access", "dataset_root")


def instantiate_dataset_split(cfg_data, split: str):
    """Instantiate a dataset for a specific split (train/val/test)."""
    shared = {k: cfg_data[k] for k in DATASET_SHARED_KEYS if k in cfg_data}
    return instantiate(cfg_data[split], **shared)


def collate_fn(states):
    m_keys = {"__key__", "__url__", "__local_path__"}
    return {
        k: (
            [s[k] for s in states]
            if k == "scenario"
            else np.stack([np.asarray(s[k]) for s in states], axis=0)
        )
        for k in states[0]
        if k not in m_keys
    }


def num_batches(dataset, batch_size: int, drop_remainder: bool = False) -> int:
    n = len(dataset)
    if drop_remainder:
        return max(1, n // batch_size) if n else 0
    return max(1, math.ceil(n / batch_size)) if n else 0


def _ensure_grain_flags_parsed(*_args) -> None:
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()


def _resolve_worker_count(worker_count):
    if worker_count is None:
        return os.cpu_count() or 4
    return int(worker_count)


class GrainLoader:
    def __init__(self, dataset, num_batches: int):
        self._dataset = dataset
        self._num_batches = num_batches

    def __iter__(self):
        return iter(self._dataset)

    def __len__(self):
        return self._num_batches


def build_grain_dataloader(dataset, cfg) -> GrainLoader:
    _ensure_grain_flags_parsed()
    batch_size = int(cfg.batch_size)
    drop_remainder = bool(cfg.get("drop_remainder", False))
    processed = WebDatasetGrainSource(dataset).batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        batch_fn=collate_fn,
    )
    worker_count = _resolve_worker_count(cfg.get("worker_count", 0))
    if worker_count > 0:
        processed = processed.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=worker_count,
                enable_profiling=False,
            ),
            worker_init_fn=_ensure_grain_flags_parsed,
        )
    return GrainLoader(processed, num_batches(dataset, batch_size, drop_remainder))


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_data, cfg_dl, **kwargs):
        super().__init__()
        self.cfg_data = cfg_data
        self.cfg_dl = cfg_dl
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _dataset(self, split):
        return instantiate_dataset_split(self.cfg_data, split)

    def setup(self, stage):
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("val")
        elif stage == "test":
            self.test_dataset = self._dataset("test")
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def train_dataloader(self):
        if self.cfg_dl.get("train", None) is None:
            return None
        return build_grain_dataloader(self.train_dataset, self.cfg_dl.train)

    def val_dataloader(self):
        if self.cfg_dl.get("val", None) is None:
            return None
        return build_grain_dataloader(self.val_dataset, self.cfg_dl.val)

    def test_dataloader(self):
        if self.cfg_dl.get("test", None) is None:
            return None
        return build_grain_dataloader(self.test_dataset, self.cfg_dl.test)
