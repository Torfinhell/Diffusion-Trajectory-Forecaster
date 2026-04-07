import dataclasses
import logging
import pickle
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from waymax import config, dataloader

from .data_process import data_process_scenarios


LOGGER = logging.getLogger(__name__)
CHUNKED_DATASET_FORMAT = "diffusion_tracker_chunked_v1"


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


def iter_processed_samples(
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
        yield processed


def save_processed_samples(processed_path, samples, flush_every=512):
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    if flush_every <= 0:
        raise ValueError("flush_every must be a positive integer.")

    chunk_dir = processed_path.parent / f"{processed_path.name}.chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk = []
    chunk_paths = []
    chunk_sizes = []
    chunk_idx = 0
    total_samples = 0

    def _flush_chunk(items):
        nonlocal chunk_idx, total_samples
        chunk_name = f"chunk_{chunk_idx:06d}.pkl"
        chunk_path = chunk_dir / chunk_name
        with chunk_path.open("wb") as file:
            pickle.dump(items, file)
        chunk_paths.append(str(chunk_path.relative_to(processed_path.parent)))
        chunk_sizes.append(len(items))
        total_samples += len(items)
        chunk_idx += 1

    for sample in samples:
        chunk.append(sample)
        if len(chunk) >= flush_every:
            _flush_chunk(chunk)
            chunk = []

    if chunk:
        _flush_chunk(chunk)

    manifest = {
        "format": CHUNKED_DATASET_FORMAT,
        "chunks": chunk_paths,
        "chunk_sizes": chunk_sizes,
        "num_samples": total_samples,
    }
    with processed_path.open("wb") as file:
        pickle.dump(manifest, file)
    return processed_path
