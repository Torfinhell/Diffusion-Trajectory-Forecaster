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
        gt_mask: jnp.ndarray | None,
    ) -> None:
        agents, flat_dim = pred_xy.shape
        H = flat_dim // 2

        pred = pred_xy.reshape(agents, H, 2)
        gt = gt_xy.reshape(agents, H, 2)

        diff = pred - gt
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist_last = dist[..., -1]
        mask = gt_mask.reshape(agents, H, 2)[..., 0]
        mask_last = mask[..., -1]

        self.state.sum_error += jnp.sum(dist_last * mask_last.astype(jnp.float32))
        self.state.count += jnp.sum(mask_last.astype(jnp.float32))
