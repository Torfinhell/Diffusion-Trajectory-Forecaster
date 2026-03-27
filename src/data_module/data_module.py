import jax.numpy as jnp
import pytorch_lightning as L
from hydra.utils import instantiate
from jax.tree_util import tree_map

from src.data_module.dataset import DiffusionTrackerDataset


def tree_collate(states):
    return tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_data, cfg_dl, **kwargs):
        super().__init__()
        self.cfg_data = cfg_data
        self.cfg_dl = cfg_dl

    def _dataset(self, split):
        split_cfg = self.cfg_data[split]
        return DiffusionTrackerDataset(
            processed_path=split_cfg.processed_path,
            dvc_file=split_cfg.dvc_file,
        )

    def setup(self, stage):
        if stage in (None, "fit"):
            self.val_dataset = self._dataset("val")
        elif stage == "test":
            self.test_dataset = self._dataset("test")
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def train_dataloader(self):
        self.train_dataset = self._dataset("train")
        return instantiate(
            self.cfg_dl.train,
            collate_fn=tree_collate,
            dataset=self.train_dataset,
        )

    def val_dataloader(self):
        return instantiate(
            self.cfg_dl.val,
            collate_fn=tree_collate,
            dataset=self.val_dataset,
        )

    def test_dataloader(self):
        return instantiate(
            self.cfg_dl.test,
            collate_fn=tree_collate,
            dataset=self.test_dataset,
        )
