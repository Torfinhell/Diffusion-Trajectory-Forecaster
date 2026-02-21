from copy import deepcopy

import lightning as L
import torch


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    FCOS requires images as a list and targets as a list of dicts.
    Handles variable-sized bounding boxes across batch.
    """

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
    ):
        super().__init__()

    def setup(self, stage):
        if stage == "fit":
            pass
        if stage == "predict":
            pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
