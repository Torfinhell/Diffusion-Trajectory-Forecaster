from collections.abc import Mapping

import numpy as np
import pytorch_lightning as L
from hydra.utils import instantiate

from src.data_module.wb_dataset import Dataset


def tree_collate(states):
    sample = states[0]

    if any(state is None for state in states):
        return list(states)

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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _dataset(self, split):
        split_cfg = self.cfg_data[split]
        return Dataset.build_webdataset(
            split=split,
            split_cfg=split_cfg,
            is_train=(split == "train"),
        )

    @staticmethod
    def _loader_cfg_dict(loader_cfg):
        return {key: value for key, value in loader_cfg.items()}

    def setup(self, stage):
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("val")
        elif stage == "test":
            self.test_dataset = self._dataset("test")
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        del device, dataloader_idx
        return batch

    def train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = self._dataset("train")
        loader_cfg = self._loader_cfg_dict(self.cfg_dl.train)
        loader_cfg.pop("chunk_sampler", None)
        loader_cfg.pop("shuffle", None)

        return instantiate(
            loader_cfg,
            collate_fn=tree_collate,
            dataset=self.train_dataset,
        )

    def val_dataloader(self):
        loader_cfg = self._loader_cfg_dict(self.cfg_dl.val)
        loader_cfg.pop("shuffle", None)
        return instantiate(
            loader_cfg,
            collate_fn=tree_collate,
            dataset=self.val_dataset,
        )

    def test_dataloader(self):
        loader_cfg = self._loader_cfg_dict(self.cfg_dl.test)
        loader_cfg.pop("shuffle", None)
        return instantiate(
            loader_cfg,
            collate_fn=tree_collate,
            dataset=self.test_dataset,
        )
