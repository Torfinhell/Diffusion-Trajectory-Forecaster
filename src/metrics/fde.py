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
        agents_coeffs: jnp.ndarray | None,
    ) -> None:
        diff = pred_xy - gt_xy
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist_last = dist[..., -1]
        if agents_coeffs is None:
            weights = jnp.ones(dist_last.shape, dtype=jnp.float32)
        else:
            weights = jnp.asarray(agents_coeffs, dtype=jnp.float32)

        self.state.sum_error += jnp.sum(dist_last * weights)
        self.state.count += jnp.sum(weights)
