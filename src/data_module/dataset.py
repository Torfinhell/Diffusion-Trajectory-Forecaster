import dataclasses
from typing import Optional

from torch.utils.data import Dataset
from tqdm.auto import tqdm
from waymax import dataloader
from waymax.config import DatasetConfig

NUM_ZEROS = 5


class DiffusionTrackerDataset(Dataset):
    def __init__(
        self,
        waymax_config: DatasetConfig,
        gs_path: str,
        max_num_states: Optional[int] = None,
        download: Optional[
            bool
        ] = True,  # TODO download states to disk? Do we need to delete them afterwards for less memory
        start_ind: int = 0,
        max_num_objects: Optional[int] = None,
    ):
        super().__init__()
        waymax_config = waymax_config
        self.states = []
        cfg = dataclasses.replace(
            waymax_config,
            path=f"{gs_path}{str(start_ind).zfill(NUM_ZEROS)}",
            max_num_objects=max_num_objects,
        )
        it = dataloader.simulator_state_generator(config=cfg)
        for ind in tqdm(
            range(max_num_states), total=max_num_states, desc="Downloading states..."
        ):
            try:
                state = next(it)
                self.states.append(state)
            except Exception as e:
                print(
                    f"Iteration through states finished at ind:{ind} with start ind: {start_ind}, and max number of  states: {max_num_states}"
                )

    def __getitem__(self, key):
        return self.states[key]

    def __len__(self):
        return len(self.states)
