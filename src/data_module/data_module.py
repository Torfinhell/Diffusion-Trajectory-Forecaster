import jax.numpy as jnp
import numpy as np
import pytorch_lightning as L
from hydra.utils import instantiate
from jax.tree_util import tree_map
from torch.utils.data import default_collate


def tree_collate(states):
    return tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_ds, cfg_dl, **kwargs):
        super().__init__()
        self.cfg_ds = cfg_ds
        self.cfg_dl = cfg_dl

    def setup(self, stage):
        if stage == "fit":
            if getattr(self.cfg_ds, "val", None) is not None:
                self.val_dataset = instantiate(self.cfg_ds.val)
        elif stage == "test":
            if getattr(self.cfg_ds, "test", None) is not None:
                self.test_dataset = instantiate(self.cfg_ds.test)
        else:
            raise NotImplementedError("Didnt implement not fit or test stage")

    def train_dataloader(self):
        if self.cfg_ds.get("train") is not None:  # safe for both dict and DictConfig
            self.train_dataset = instantiate(self.cfg_ds.train)
        else:
            raise RuntimeError("Training dataset config not found!")
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
