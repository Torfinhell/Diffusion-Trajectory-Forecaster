import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_module.wb_dataset import Dataset
from src.models import DiffAttention


def load_sample(feat_extract_cfg_path: str, split: str, sample_idx: int):
    cfg = OmegaConf.load(feat_extract_cfg_path)
    split_cfg = cfg.data[split]
    dataset = Dataset.build_webdataset(split=split, split_cfg=split_cfg, is_train=False)
    iterator = iter(dataset)
    sample = None
    for _ in range(sample_idx + 1):
        sample = next(iterator)
    return sample


def build_model(model_cfg_path: str):
    model_cfg = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=True)
    model_cfg.pop("_target_", None)
    return DiffAttention(**model_cfg, key=jr.PRNGKey(0))


def to_jax_sample(sample: dict):
    jax_sample = {}
    for key, value in sample.items():
        np_value = np.asarray(value)
        if np.issubdtype(np_value.dtype, np.number) or np.issubdtype(np_value.dtype, np.bool_):
            jax_sample[key] = jnp.asarray(np_value)
    return jax_sample


def main():
    parser = argparse.ArgumentParser(description="Smoke-test DiffAttention shapes on one real sample.")
    parser.add_argument(
        "--feat-extract-cfg",
        default="src/configs/feat_extract/small_no_scenes.yaml",
    )
    parser.add_argument(
        "--model-cfg",
        default="src/configs/model/diffusion_attn.yaml",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    sample = load_sample(args.feat_extract_cfg, args.split, args.sample_idx)
    model = build_model(args.model_cfg)

    sample = to_jax_sample(sample)
    cond = model.prepare_conditioning(sample)
    gt_xy = sample["agent_future"][..., :2]
    t_noise = jnp.array(0.5, dtype=jnp.float32)
    x_t = jnp.zeros_like(gt_xy)
    pred = model(t_noise, x_t, cond)

    print("Loaded sample fields:")
    for key, value in sample.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    print("\nConditioning fields:")
    for key, value in cond.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    print("\nForward pass:")
    print(f"  x_t shape: {x_t.shape}")
    print(f"  pred shape: {pred.shape}")
    print(f"  expected shape: {gt_xy.shape}")
    if pred.shape != gt_xy.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match expected {gt_xy.shape}")

    print("\nDiffAttention smoke test passed.")


if __name__ == "__main__":
    main()
