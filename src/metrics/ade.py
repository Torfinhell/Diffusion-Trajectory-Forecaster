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

    def compute(self) -> jnp.ndarray:
        return self.state.sum_error / (self.state.count + self.eps)

    def update(
        self,
        pred_xy: jnp.ndarray,  # shape: (agents, H, 2)
        gt_xy: jnp.ndarray,  # shape: (agents, H, 2)
        agents_coeffs: jnp.ndarray | None,  # shape: (agents,)
    ) -> None:
        diff = pred_xy - gt_xy
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        if agents_coeffs is None:
            weights = jnp.ones(gt_xy.shape[:-1], dtype=jnp.float32)
        else:
            weights = jnp.asarray(agents_coeffs, dtype=jnp.float32)[..., None]
        self.state.sum_error += jnp.sum(dist * weights)
        self.state.count += jnp.sum(weights)
