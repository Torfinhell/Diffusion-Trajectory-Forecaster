from typing import Optional

from torch.utils.data import Dataset


class DiffusionTrackerDataset(Dataset):
    def __init__(
        self,
        waymax_config,
        gs_path: str,
        max_num_states: Optional[int] = None,
        max_num_objects: Optional[int] = None,
        download: bool = False,
        gcloud_token: Optional[str] = None,
    ):
        super().__init__()

    def __getitem__(self, key):
        pass

    def __len__(self):
        pass
