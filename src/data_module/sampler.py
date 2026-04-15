import math
import random
from collections.abc import Iterator

from torch.utils.data import BatchSampler


class ChunkWindowBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        active_chunk_window: int = 1,
        chunk_window_step: int | None = None,
        shuffle: bool = True,
        shuffle_within_window: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if active_chunk_window <= 0:
            raise ValueError("active_chunk_window must be positive")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.active_chunk_window = int(active_chunk_window)
        self.chunk_window_step = (
            int(chunk_window_step)
            if chunk_window_step is not None
            else self.active_chunk_window
        )
        if self.chunk_window_step <= 0:
            raise ValueError("chunk_window_step must be positive")

        self.shuffle = bool(shuffle)
        self.shuffle_within_window = bool(shuffle_within_window)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        chunk_ids = list(range(self.dataset.num_chunks))
        if self.shuffle:
            rng.shuffle(chunk_ids)

        pending: list[int] = []
        num_chunks = len(chunk_ids)

        for start in range(0, num_chunks, self.chunk_window_step):
            window_chunk_ids = chunk_ids[start : start + self.active_chunk_window]
            window_indices: list[int] = []
            for chunk_idx in window_chunk_ids:
                chunk_start, chunk_end = self.dataset.chunk_bounds(chunk_idx)
                window_indices.extend(range(chunk_start, chunk_end))
            if self.shuffle_within_window:
                rng.shuffle(window_indices)

            pending.extend(window_indices)
            while len(pending) >= self.batch_size:
                yield pending[: self.batch_size]
                pending = pending[self.batch_size :]

        if pending and not self.drop_last:
            yield pending

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)
