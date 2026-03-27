import dataclasses
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from waymax import config, dataloader

from .data_process import data_process_scenarios


LOGGER = logging.getLogger(__name__)


def _strip_leading_batch_axis(sample):
    cleaned = {}
    for key, value in sample.items():
        if key == "scenario":
            cleaned[key] = value
            continue
        arr = jnp.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == 1:
            cleaned[key] = jnp.squeeze(arr, axis=0)
        else:
            cleaned[key] = value
    return cleaned


def build_waymax_iterator(raw_data_url, waymax_conf_version, max_num_objects):
    waymax_config = getattr(config, waymax_conf_version)
    waymax_config = dataclasses.replace(
        waymax_config,
        path=str(raw_data_url),
        max_num_objects=max_num_objects,
    )
    return dataloader.simulator_state_generator(config=waymax_config)


def create_processed_samples(
    raw_data_url,
    waymax_conf_version,
    num_states,
    max_num_objects,
    extract_scene,
    preprocess_kwargs,
):
    iterator = build_waymax_iterator(
        raw_data_url=raw_data_url,
        waymax_conf_version=waymax_conf_version,
        max_num_objects=max_num_objects,
    )
    samples = []
    for index in tqdm(range(num_states), total=num_states, desc="Creating dataset"):
        try:
            state = next(iterator)
        except Exception as exc:
            LOGGER.warning(
                "Stopped dataset creation after %s/%s states: %s",
                index,
                num_states,
                exc,
            )
            break

        batched_scenario = jax.tree_util.tree_map(lambda x: x[None, ...], state)
        processed = _strip_leading_batch_axis(
            data_process_scenarios(batched_scenario, **preprocess_kwargs)
        )
        if extract_scene:
            processed = {"scenario": state, **processed}
        samples.append(processed)
    return samples


def save_processed_samples(processed_path, samples):
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("wb") as file:
        pickle.dump(samples, file)
    return processed_path
