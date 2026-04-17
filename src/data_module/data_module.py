from collections.abc import Mapping

import numpy as np
import pytorch_lightning as L
from hydra.utils import instantiate

from src.data_module.dataset import build_webdataset
from src.data_module.legacy_dataset import DiffusionTrackerDataset
from src.data_module.sampler import ChunkWindowBatchSampler


def tree_collate(states):
    sample = states[0]

    if isinstance(sample, Mapping):
        metadata_keys = {"__key__", "__url__", "__local_path__"}
        return {
            key: tree_collate([state[key] for state in states])
            for key in sample
            if key not in metadata_keys
        }

    if isinstance(sample, tuple):
        return tuple(tree_collate([state[idx] for state in states]) for idx in range(len(sample)))

    if isinstance(sample, list):
        return [tree_collate([state[idx] for state in states]) for idx in range(len(sample))]

    if sample is None:
        return None

    try:
        return np.stack([np.asarray(state) for state in states], axis=0)
    except Exception:
        return list(states)

class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_data, cfg_dl, **kwargs):
        super().__init__()
        self.cfg_data = cfg_data
        self.cfg_dl = cfg_dl

    def _dataset(self, split):
        split_cfg = self.cfg_data[split]
        storage_format = str(split_cfg.get("storage_format", "")).lower()
        if storage_format in {"webdataset", "wds", "tar_shards"}:
            return build_webdataset(split_cfg=split_cfg, is_train=(split == "train"))
        return DiffusionTrackerDataset(
            processed_path=split_cfg.processed_path,
            dvc_file=split_cfg.dvc_file,
            chunk_cache_size=split_cfg.get("chunk_cache_size", 8),
        )

    @staticmethod
    def _loader_cfg_dict(loader_cfg):
        return {key: value for key, value in loader_cfg.items()}

    def setup(self, stage):
        if stage in (None, "fit"):
            self.val_dataset = self._dataset("val")
        elif stage == "test":
            self.test_dataset = self._dataset("test")
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        del device, dataloader_idx
        return batch

    def train_dataloader(self):
        self.train_dataset = self._dataset("train")
        loader_cfg = self._loader_cfg_dict(self.cfg_dl.train)
        sampler_cfg = loader_cfg.pop("chunk_sampler", None)
        storage_format = str(self.cfg_data.train.get("storage_format", "")).lower()

        if storage_format in {"webdataset", "wds", "tar_shards"}:
            loader_cfg.pop("shuffle", None)
            return instantiate(
                loader_cfg,
                collate_fn=tree_collate,
                dataset=self.train_dataset,
            )

        if sampler_cfg and sampler_cfg.get("enabled", False):
            batch_size = int(loader_cfg.pop("batch_size"))
            shuffle = bool(loader_cfg.pop("shuffle", False))
            batch_sampler = ChunkWindowBatchSampler(
                self.train_dataset,
                batch_size=batch_size,
                active_chunk_window=int(sampler_cfg.get("active_chunk_window", 1)),
                chunk_window_step=(
                    int(sampler_cfg["chunk_window_step"])
                    if sampler_cfg.get("chunk_window_step") is not None
                    else None
                ),
                shuffle=shuffle,
                shuffle_within_window=bool(
                    sampler_cfg.get("shuffle_within_window", True)
                ),
                drop_last=bool(loader_cfg.pop("drop_last", False)),
                seed=int(sampler_cfg.get("seed", 0)),
            )
            return instantiate(
                loader_cfg,
                collate_fn=tree_collate,
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
            )

        return instantiate(
            loader_cfg,
            collate_fn=tree_collate,
            dataset=self.train_dataset,
        )

    def val_dataloader(self):
        loader_cfg = self._loader_cfg_dict(self.cfg_dl.val)
        storage_format = str(self.cfg_data.val.get("storage_format", "")).lower()
        if storage_format in {"webdataset", "wds", "tar_shards"}:
            loader_cfg.pop("shuffle", None)
            return instantiate(
                loader_cfg,
                collate_fn=tree_collate,
                dataset=self.val_dataset,
            )
        return instantiate(
            loader_cfg,
            collate_fn=tree_collate,
            dataset=self.val_dataset,
        )

    def test_dataloader(self):
        loader_cfg = self._loader_cfg_dict(self.cfg_dl.test)
        storage_format = str(self.cfg_data.test.get("storage_format", "")).lower()
        if storage_format in {"webdataset", "wds", "tar_shards"}:
            loader_cfg.pop("shuffle", None)
            return instantiate(
                loader_cfg,
                collate_fn=tree_collate,
                dataset=self.test_dataset,
            )
        return instantiate(
            loader_cfg,
            collate_fn=tree_collate,
            dataset=self.test_dataset,
        )
