import dataclasses
import pickle
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from waymax import config, dataloader

from .data_process import data_process_scenarios


class DiffusionTrackerDataset(Dataset):
    GLOBAL_ITER = None
    TRAIN_DIR = 0

    def __init__(
        self,
        waymax_conf_version: str,
        gs_path: str,
        num_states: Optional[int] = None,
        download_folder: Optional[str] = None,
        max_num_objects: Optional[int] = None,
        extract_scene=False,
        extract_data_conf=None,
    ):
        super().__init__()
        if self.__class__.GLOBAL_ITER is None:
            waymax_config = getattr(config, waymax_conf_version)
            waymax_config = dataclasses.replace(
                waymax_config,
                path=f"{gs_path}",
                max_num_objects=max_num_objects,
            )
            self.__class__.GLOBAL_ITER = dataloader.simulator_state_generator(
                config=waymax_config
            )
        self.data = []
        if download_folder is not None:
            download_folder = Path(download_folder)
            if "train" in str(download_folder):
                download_path = (
                    download_folder / f"states_{self.__class__.TRAIN_DIR}.pkl"
                )
                self.__class__.TRAIN_DIR += 1
            else:
                download_path = download_folder / "states.pkl"
            download_path.parent.mkdir(parents=True, exist_ok=True)
        if download_folder is not None and download_path.exists():
            with open(download_path, "rb") as file:
                self.data = pickle.load(file)
            # if not extract_scene:
            #     for i, state in enumerate(self.data):
            #         batched_scenario = jax.tree_util.tree_map(
            #             lambda x: x[None, ...], state["scenario"]
            #         )
            #         self.data[i] = data_process_scenarios(
            #             batched_scenario, **extract_data_conf
            #         )
            #         self.data[i].update(state)
            print(f"Downloaded states from {download_path}")
        else:
            for ind in tqdm(
                range(num_states), total=num_states, desc="Downloading states..."
            ):
                try:
                    state = next(self.__class__.GLOBAL_ITER)
                    batched_scenario = jax.tree_util.tree_map(
                        lambda x: x[None, ...], state
                    )
                    if not extract_scene:
                        self.data.append(
                            data_process_scenarios(
                                batched_scenario, **extract_data_conf
                            )
                        )
                    else:
                        self.data.append({"scenario": state})
                        self.data[-1].update(
                            data_process_scenarios(
                                batched_scenario, **extract_data_conf
                            )
                        )

                except Exception as e:
                    print(
                        f"Iteration through states finished with exception {e} \n"
                        f"at ind: {ind}, and max number of states: {num_states} \n"
                        f"gs_path is {gs_path}"
                    )
                    break
            if download_folder is not None:
                with open(download_path, "wb") as file:
                    pickle.dump(self.data, file)
                print(f"Downloaded states to {download_path}")

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
