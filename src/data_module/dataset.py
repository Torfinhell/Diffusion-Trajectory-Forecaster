import pickle
import subprocess
from pathlib import Path

from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class DiffusionTrackerDataset(Dataset):
    def __init__(self, processed_path: str, dvc_file: str | None = None):
        super().__init__()
        dataset_path = Path(to_absolute_path(processed_path))
        if not dataset_path.exists():
            if dvc_file is None:
                raise FileNotFoundError(
                    f"Processed dataset not found at {dataset_path} and no dvc_file was provided."
                )
            print(f"Processed dataset not found at {dataset_path}. Pulling with DVC.")
            try:
                subprocess.run(["dvc", "pull", dvc_file], check=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Could not run `dvc pull` because the `dvc` CLI is not installed."
                ) from exc

        with dataset_path.open("rb") as file:
            self.data = pickle.load(file)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
