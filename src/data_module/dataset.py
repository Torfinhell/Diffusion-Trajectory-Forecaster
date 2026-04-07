import bisect
import pickle
import subprocess
from pathlib import Path

from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from src.data_module.dataset_creation import CHUNKED_DATASET_FORMAT


class DiffusionTrackerDataset(Dataset):
    def __init__(self, processed_path: str, dvc_file: str | None = None):
        super().__init__()
        self._chunk_cache = {}
        self._chunk_paths = []
        self._chunk_sizes = []
        self._chunk_ends = []
        self._length = 0

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
            loaded = pickle.load(file)

        if isinstance(loaded, list):
            self._mode = "legacy"
            self.data = loaded
            self._length = len(self.data)
            return

        if isinstance(loaded, dict) and loaded.get("format") == CHUNKED_DATASET_FORMAT:
            self._mode = "chunked"
            self._chunk_paths = [dataset_path.parent / p for p in loaded.get("chunks", [])]
            self._chunk_sizes = loaded.get("chunk_sizes", [])
            self._length = int(loaded.get("num_samples", 0))

            if len(self._chunk_paths) != len(self._chunk_sizes):
                raise RuntimeError("Corrupted chunked dataset manifest: chunk metadata mismatch.")

            cumulative = 0
            for size in self._chunk_sizes:
                cumulative += size
                self._chunk_ends.append(cumulative)

            if self._length != cumulative:
                raise RuntimeError("Corrupted chunked dataset manifest: num_samples mismatch.")
            return

        raise RuntimeError(
            "Unsupported dataset format in processed file. Recreate dataset artifacts."
        )

    def _load_chunk(self, chunk_idx: int):
        chunk = self._chunk_cache.get(chunk_idx)
        if chunk is not None:
            return chunk

        with self._chunk_paths[chunk_idx].open("rb") as file:
            chunk = pickle.load(file)

        # Keep a single hot chunk in memory to stay memory-safe during training.
        self._chunk_cache = {chunk_idx: chunk}
        return chunk

    def __getitem__(self, key):
        idx = int(key)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("dataset index out of range")

        if self._mode == "legacy":
            return self.data[idx]

        chunk_idx = bisect.bisect_right(self._chunk_ends, idx)
        chunk_start = 0 if chunk_idx == 0 else self._chunk_ends[chunk_idx - 1]
        in_chunk_idx = idx - chunk_start
        chunk = self._load_chunk(chunk_idx)
        return chunk[in_chunk_idx]

    def __len__(self):
        return self._length
