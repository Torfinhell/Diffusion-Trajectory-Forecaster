import dataclasses
import json
import logging
import pickle
import shutil
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from waymax import config, dataloader

from .data_process import data_process_scenarios


LOGGER = logging.getLogger(__name__)
CHUNKED_DATASET_FORMAT = "diffusion_tracker_chunked_v1"
SINGLE_FILE_CHUNKED_DATASET_FORMAT = "diffusion_tracker_single_file_chunked_v1"
WEBDATASET_FORMAT = "diffusion_tracker_webdataset_v1"
SINGLE_FILE_INDEX_MAGIC = b"DTDSIDX1"
SINGLE_FILE_INDEX_TRAILER = struct.Struct("<Q8s")
WEBDATASET_INDEX_FILENAME = "index.json"


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
    start_index,
    max_num_objects,
    extract_scene,
    preprocess_kwargs,
):
    iterator = build_waymax_iterator(
        raw_data_url=raw_data_url,
        waymax_conf_version=waymax_conf_version,
        max_num_objects=max_num_objects,
    )
    total_to_read = int(start_index) + int(num_states)
    produced = 0
    for index in tqdm(range(total_to_read), total=total_to_read, desc="Creating dataset"):
        try:
            state = next(iterator)
        except Exception as exc:
            if index == 0:
                raise RuntimeError(
                    "Dataset creation failed before producing the first sample. "
                    f"raw_data_url={raw_data_url}, waymax_conf_version={waymax_conf_version}"
                ) from exc
            LOGGER.warning(
                "Stopped dataset creation after %s/%s states: %s",
                index,
                num_states,
                exc,
            )
            break

        if index < int(start_index):
            continue

        batched_scenario = jax.tree_util.tree_map(lambda x: x[None, ...], state)
        processed = _strip_leading_batch_axis(
            data_process_scenarios(batched_scenario, **preprocess_kwargs)
        )
        if extract_scene:
            processed = {"scenario": state, **processed}
        yield processed
        produced += 1
        if produced >= int(num_states):
            break


def resolve_webdataset_output_root(processed_path) -> Path:
    processed_path = Path(processed_path)
    suffix = processed_path.suffix or ".data"
    return processed_path.with_suffix(f"{suffix}.wds")


def _save_processed_samples_as_directory_chunks(processed_path, samples, flush_every):
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
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


def _save_processed_samples_as_single_file(processed_path, samples, flush_every):
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = processed_path.parent / f"{processed_path.name}.chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    chunk = []
    chunk_offsets = []
    chunk_sizes = []
    total_samples = 0

    with processed_path.open("wb") as file:

        def _flush_chunk(items):
            nonlocal total_samples
            chunk_offsets.append(file.tell())
            pickle.dump(items, file, protocol=pickle.HIGHEST_PROTOCOL)
            chunk_sizes.append(len(items))
            total_samples += len(items)
            file.flush()

        for sample in samples:
            chunk.append(sample)
            if len(chunk) >= flush_every:
                _flush_chunk(chunk)
                chunk = []

        if chunk:
            _flush_chunk(chunk)

        index_offset = file.tell()
        index = {
            "format": SINGLE_FILE_CHUNKED_DATASET_FORMAT,
            "chunk_offsets": chunk_offsets,
            "chunk_sizes": chunk_sizes,
            "num_samples": total_samples,
        }
        pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.write(SINGLE_FILE_INDEX_TRAILER.pack(index_offset, SINGLE_FILE_INDEX_MAGIC))
        file.flush()

    return processed_path


def _save_processed_samples_as_webdataset(processed_path, samples, flush_every):
    try:
        import webdataset as wds
    except ImportError as exc:
        raise RuntimeError(
            "storage_format='webdataset' requires the 'webdataset' package to be installed."
        ) from exc

    output_root = resolve_webdataset_output_root(processed_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(output_root / "shard-%06d.tar")
    total_samples = 0

    with wds.ShardWriter(shard_pattern, maxcount=flush_every) as sink:
        for index, sample in enumerate(samples):
            sink.write(
                {
                    "__key__": f"{index:09d}",
                    "pth": wds.torch_dumps(sample),
                }
            )
            total_samples += 1

    index_path = output_root / WEBDATASET_INDEX_FILENAME
    metadata = {
        "format": WEBDATASET_FORMAT,
        "num_samples": total_samples,
        "num_shards": len(list(output_root.glob("shard-*.tar"))),
        "shard_glob": "shard-*.tar",
    }
    index_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_root


def save_processed_samples(
    processed_path,
    samples,
    flush_every=512,
    storage_format="directory_chunks",
):
    if flush_every <= 0:
        raise ValueError("flush_every must be a positive integer.")

    storage_format = str(storage_format).lower()
    if storage_format in {"directory_chunks", "chunked", "chunks"}:
        return _save_processed_samples_as_directory_chunks(
            processed_path,
            samples,
            flush_every,
        )
    if storage_format in {"single_file", "single_pkl", "streamed_single_file"}:
        return _save_processed_samples_as_single_file(
            processed_path,
            samples,
            flush_every,
        )
    if storage_format in {"webdataset", "wds", "tar_shards"}:
        return _save_processed_samples_as_webdataset(
            processed_path,
            samples,
            flush_every,
        )
    raise ValueError(
        "Unsupported storage_format. "
        "Use one of: directory_chunks, chunked, chunks, single_file, single_pkl, "
        "streamed_single_file, webdataset, wds, tar_shards."
    )
