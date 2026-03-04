import jax.numpy as jnp
import lightning as L
from hydra.utils import instantiate
from jax import tree_util

from src.data_module.dataset import DiffusionTrackerDataset


def waymax_collate(states):
    return tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


# def numpy_collate(batch):
#     if isinstance(batch[0], np.ndarray):
#         return np.stack(batch)
#     elif isinstance(batch[0], (tuple, list)):
#         transposed = zip(*batch)
#         return [numpy_collate(samples) for samples in transposed]
#     else:
#         return np.array(batch)


class DiffusionTrackerDataModule(L.LightningDataModule):
    def __init__(self, cfg_data_module, **kwargs):
        super().__init__()
        self.cfg_data_module = cfg_data_module

    def setup(self, stage):
        if stage == "fit":
            if self.train_ds_cfg is not None:
                self.train_dataset = instantiate(self.cfg_data_module.train_ds_cfg)
            if self.val_ds_cfg is not None:
                self.val_dataset = instantiate(self.cfg_data_module.val_ds_cfg)
            if self.test_ds_cfg is not None:
                self.test_dataset = instantiate(self.cfg_data_module.test_ds_cfg)
        else:
            raise NotImplementedError("Didnt implement not fit stage")

    def train_dataloader(self):
        return instantiate(
            self.cfg_data_module.train_dl_cfg,
            collate_fn=waymax_collate,
            dataset=self.train_dataset,
        )

    def val_dataloader(self):
        return instantiate(
            self.cfg_data_module.val_dl_cfg,
            collate_fn=waymax_collate,
            dataset=self.val_dataset,
        )

    def test_dataloader(self):
        return instantiate(
            self.cfg_data_module.test_dl_cfg,
            collate_fn=waymax_collate,
            dataset=self.test_dataset,
        )
