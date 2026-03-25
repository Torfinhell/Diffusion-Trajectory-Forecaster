from dataclasses import dataclass

import jax.numpy as jnp

from .base import BaseMetric, temporal_valid_mask


@dataclass
class FDEState:
    sum_error: jnp.ndarray
    count: jnp.ndarray


class FdeMetric(BaseMetric):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reset()

    def reset(self):
        self.state = FDEState(
            sum_error=jnp.array(0.0, jnp.float32), count=jnp.array(0.0, jnp.float32)
        )

    def compute(self) -> jnp.ndarray:
        return self.state.sum_error / (self.state.count + self.eps)

    def update(
        self,
        pred_xy: jnp.ndarray,
        gt_xy: jnp.ndarray,
        gt_mask: jnp.ndarray | None,
    ) -> None:
        diff = pred_xy - gt_xy
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        mask = temporal_valid_mask(gt_xy, gt_mask)
        time_idx = jnp.arange(mask.shape[-1], dtype=jnp.int32)
        last_valid_idx = jnp.max(jnp.where(mask, time_idx, -1), axis=-1)
        has_valid_timestep = last_valid_idx >= 0
        safe_last_valid_idx = jnp.maximum(last_valid_idx, 0)
        dist_last = jnp.take_along_axis(
            dist, safe_last_valid_idx[..., None], axis=-1
        ).squeeze(axis=-1)
        weight = has_valid_timestep.astype(jnp.float32)

        self.state.sum_error += jnp.sum(dist_last * weight)
        self.state.count += jnp.sum(weight)
