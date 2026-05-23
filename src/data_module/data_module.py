from collections.abc import Mapping

import numpy as np
import pytorch_lightning as L
from flax.jax_utils import prefetch_to_device
from hydra.utils import instantiate

DATASET_SHARED_KEYS = ("allow_upload", "s3_url", "data_access", "dataset_root")


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


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_data, cfg_dl, **kwargs):
        super().__init__()
        self.cfg_data = cfg_data
        self.cfg_dl = cfg_dl
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _dataset(self, split):
        shared = {
            k: self.cfg_data[k] for k in DATASET_SHARED_KEYS if k in self.cfg_data
        }
        return instantiate(self.cfg_data[split], **shared)

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
        dl = instantiate(
            self.cfg_dl.train,
            collate_fn=collate_fn,
            dataset=self.train_dataset,
        )
        return prefetch_to_device(dl, size=2)

    def val_dataloader(self):
        dl = instantiate(
            self.cfg_dl.val,
            collate_fn=collate_fn,
            dataset=self.val_dataset,
        )
        return prefetch_to_device(dl, size=2)

    def test_dataloader(self):
        dl = instantiate(
            self.cfg_dl.test,
            collate_fn=collate_fn,
            dataset=self.test_dataset,
        )
        return prefetch_to_device(dl, size=2)
