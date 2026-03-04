from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class ADEState:
    sum_error: jnp.ndarray  # scalar
    count: jnp.ndarray      # scalar

def ade_init() -> ADEState:
    return ADEState(sum_error=jnp.array(0.0, jnp.float32),
                    count=jnp.array(0.0, jnp.float32))

def ade_update(state: ADEState, pred_xy: jnp.ndarray, gt_xy: jnp.ndarray, valid: jnp.ndarray | None) -> ADEState:
    # pred_xy, gt_xy: (..., T, 2)
    diff = pred_xy - gt_xy
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (..., T)

    if valid is None:
        batch_sum = jnp.sum(dist)
        batch_cnt = dist.size
        batch_cnt = jnp.array(batch_cnt, jnp.float32)
    else:
        valid_f = valid.astype(jnp.float32)          # (..., T)
        batch_sum = jnp.sum(dist * valid_f)
        batch_cnt = jnp.sum(valid_f)

    return ADEState(sum_error=state.sum_error + batch_sum,
                    count=state.count + batch_cnt)

def ade_compute(state: ADEState, eps: float = 1e-8) -> jnp.ndarray:
    return state.sum_error / (state.count + eps)


class ADE:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.state = ade_init()

    def update(self, pred_xy: jnp.ndarray, gt_xy: jnp.ndarray, valid: jnp.ndarray | None = None):
        self.state = ade_update(self.state, pred_xy, gt_xy, valid)

    def compute(self) -> jnp.ndarray:
        return ade_compute(self.state, self.eps)

        