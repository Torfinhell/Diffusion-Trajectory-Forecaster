from dataclasses import dataclass

import jax.numpy as jnp

from .base import BaseMetric


@dataclass
class ADEState:
    sum_error: jnp.ndarray  # scalar
    count: jnp.ndarray  # scalar


class AdeMetric(BaseMetric):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reset()

    def reset(self):
        self.state = ADEState(
            sum_error=jnp.array(0.0, jnp.float32), count=jnp.array(0.0, jnp.float32)
        )

    def compute(state: ADEState, eps: float = 1e-8) -> jnp.ndarray:
        return state.sum_error / (state.count + eps)

    def update(
        state: ADEState,
        pred_xy: jnp.ndarray,
        gt_xy: jnp.ndarray,
        valid: jnp.ndarray | None,
    ) -> ADEState:
        # pred_xy, gt_xy: (..., T, 2)
        diff = pred_xy - gt_xy
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (..., T)

        if valid is None:
            batch_sum = jnp.sum(dist)
            batch_cnt = dist.size
            batch_cnt = jnp.array(batch_cnt, jnp.float32)
        else:
            valid_f = valid.astype(jnp.float32)  # (..., T)
            batch_sum = jnp.sum(dist * valid_f)
            batch_cnt = jnp.sum(valid_f)
