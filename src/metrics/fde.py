from dataclasses import dataclass

import jax.numpy as jnp

from .base import BaseMetric


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
        agents_coeffs: jnp.ndarray,
        future_valid: jnp.ndarray,  # shape: (agents, H, 1)
    ) -> None:
        diff = pred_xy - gt_xy
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        valid = jnp.asarray(future_valid)[..., 0].astype(bool)
        time_idx = jnp.arange(valid.shape[-1], dtype=jnp.int32)
        last_valid_idx = jnp.max(jnp.where(valid, time_idx, -1), axis=-1)
        has_valid = last_valid_idx >= 0
        last_valid_idx = jnp.maximum(last_valid_idx, 0)
        weights = jnp.asarray(agents_coeffs, dtype=jnp.float32) * has_valid.astype(
            jnp.float32
        )

        dist_last = jnp.take_along_axis(dist, last_valid_idx[..., None], axis=-1).squeeze(-1)
        self.state.sum_error += jnp.sum(dist_last * weights)
        self.state.count += jnp.sum(weights)
