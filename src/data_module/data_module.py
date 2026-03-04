import jax.numpy as jnp
import lightning as L
import numpy as np
from hydra.utils import instantiate
from jax.tree_util import tree_map
from torch.utils.data import default_collate

from src.data_module.dataset import DiffusionTrackerDataset


def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_data_module, **kwargs):
        super().__init__()
        self.cfg_data_module = cfg_data_module

    def setup(self, stage):
        if stage == "fit":
            if self.cfg_data_module.train_ds_cfg is not None:
                self.train_dataset = instantiate(self.cfg_data_module.train_ds_cfg)
            if self.cfg_data_module.val_ds_cfg is not None:
                self.val_dataset = instantiate(self.cfg_data_module.val_ds_cfg)
            if self.cfg_data_module.test_ds_cfg is not None:
                self.test_dataset = instantiate(self.cfg_data_module.test_ds_cfg)
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def train_dataloader(self):
        return instantiate(
            self.cfg_data_module.train_dl_cfg,
            collate_fn=numpy_collate,
            dataset=self.train_dataset,
        )

    def val_dataloader(self):
        return instantiate(
            self.cfg_data_module.val_dl_cfg,
            collate_fn=numpy_collate,
            dataset=self.val_dataset,
        )

    def test_dataloader(self):
        return instantiate(
            self.cfg_data_module.test_dl_cfg,
            collate_fn=numpy_collate,
            dataset=self.test_dataset,
        )
