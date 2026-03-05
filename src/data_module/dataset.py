import dataclasses
import pickle
from pathlib import Path
from typing import Optional

from hydra.utils import instantiate
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from waymax import config, dataloader


class DiffusionTrackerDataset(Dataset):
    NUM_ZEROS = 5
    GLOBAL_ITER = None

    def __init__(
        self,
        waymax_conf_version: str,
        gs_path: str,
        num_states: Optional[int] = None,
        download_path: Optional[str] = None,
        start_ind: int = 0,
        max_num_objects: Optional[int] = None,
    ):
        super().__init__()
        if self.__class__.GLOBAL_ITER is None:
            waymax_config = getattr(config, waymax_conf_version)
            waymax_config = dataclasses.replace(
                waymax_config,
                path=f"{gs_path}{str(start_ind).zfill(self.__class__.NUM_ZEROS)}",
                max_num_objects=max_num_objects,
            )
            self.__class__.GLOBAL_ITER = dataloader.simulator_state_generator(
                config=waymax_config
            )
        self.states = []
        file_name = "states.pickle"
        if download_path is not None:
            download_path = Path(download_path)
        if download_path is not None and Path(download_path / file_name).exists():
            with open(download_path / "states.pickle", "rb") as file:
                self.states = pickle.load(file)
        else:
            for ind in tqdm(
                range(num_states), total=num_states, desc="Downloading states..."
            ):
                try:
                    state = next(self.__class__.GLOBAL_ITER)
                    self.states.append(state)
                except Exception as e:
                    inner_gs_path = (
                        f"{gs_path}{str(ind).zfill(self.__class__.NUM_ZEROS)}"
                    )
                    print(
                        f"Iteration through states finished with exception {e} \n"
                        f"at ind: {ind} with start ind: {start_ind}, and max number of states: {num_states} \n"
                        f"gs_path is {inner_gs_path}"
                    )
                    break
            if download_path is not None:
                with open(download_path / "states.pickle", "wb") as file:
                    pickle.dump(self.states, file)
                print(f"Downloaded states to {download_path}")

    def __getitem__(self, key):
        return self.states[key]

    def __len__(self):
        return len(self.states)
