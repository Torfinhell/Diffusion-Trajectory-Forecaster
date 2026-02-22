from copy import deepcopy
from functools import partial

import lightning as L
import torch
from hydra.utils import instantiate

from src.data_module.dataset import DiffusionTrackerDataset


def collate_fn(batch):

    return {
        key: (
            torch.tensor([elem[key] for elem in batch])
            if key in []
            else [elem[key] for elem in batch]
        )
        for key in batch[0]
    }


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_ds_cfg,
        val_ds_cfg,
        test_ds_cfg,
        train_dl_cfg,
        val_dl_cfg,
        test_dl_cfg,
        **kwargs
    ):
        super().__init__()
        self.train_ds_cfg = train_ds_cfg
        self.val_ds_cfg = val_ds_cfg
        self.test_ds_cfg = test_ds_cfg
        self.train_dl_cfg = train_ds_cfg
        self.val_dl_cfg = val_dl_cfg
        self.test_dl_cfg = test_dl_cfg

    def setup(self, stage):
        if stage == "fit":
            if self.train_ds_cfg is not None:
                self.train_dataset = instantiate(self.train_ds_cfg)
            if self.val_ds_cfg is not None:
                self.val_dataset = instantiate(self.val_ds_cfg)
            if self.test_ds_cfg is not None:
                self.test_dataset = instantiate(self.test_ds_cfg)
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def train_dataloader(self):
        return instantiate(
            self.train_dl_cfg,
            collate_fn=collate_fn,
            dataset=self.train_dataset,
        )

    def val_dataloader(self):
        return instantiate(
            self.val_dl_cfg,
            collate_fn=collate_fn,
            dataset=self.val_dataset,
        )

    def test_dataloader(self):
        return instantiate(
            self.test_dl_cfg,
            collate_fn=collate_fn,
            dataset=self.test_dataset,
        )
