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
        agents_coeffs: jnp.ndarray,  # shape: (agents,)
        future_valid: jnp.ndarray,  # shape: (agents, H, 1)
    ) -> None:
        diff = pred_xy - gt_xy
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        weights = jnp.asarray(agents_coeffs, dtype=jnp.float32)[..., None]
        weights = weights * jnp.asarray(future_valid, dtype=jnp.float32)[..., 0]
        self.state.sum_error += jnp.sum(dist * weights)
        self.state.count += jnp.sum(weights)
