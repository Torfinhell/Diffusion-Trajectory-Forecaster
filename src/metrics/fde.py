from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class FDEState:
    sum_error: jnp.ndarray
    count: jnp.ndarray


def fde_init() -> FDEState:
    return FDEState(
        sum_error=jnp.array(0.0, jnp.float32),
        count=jnp.array(0.0, jnp.float32),
    )


def fde_update(
    state: FDEState,
    pred_xy: jnp.ndarray,
    gt_xy: jnp.ndarray,
    valid: jnp.ndarray | None,
) -> FDEState:
    """
    FDE (Final Displacement Error)

    pred_xy : (..., T, 2)
    gt_xy   : (..., T, 2)
    valid   : (..., T) optional
    """

    pred_final = pred_xy[..., -1, :]
    gt_final = gt_xy[..., -1, :]

    diff = pred_final - gt_final
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (...)

    if valid is not None:
        final_valid = valid[..., -1].astype(jnp.float32)
        batch_sum = jnp.sum(dist * final_valid)
        batch_cnt = jnp.sum(final_valid)
    else:
        batch_sum = jnp.sum(dist)
        batch_cnt = jnp.array(dist.size, jnp.float32)

    return FDEState(
        sum_error=state.sum_error + batch_sum,
        count=state.count + batch_cnt,
    )


def fde_compute(state: FDEState, eps: float = 1e-8) -> jnp.ndarray:
    return state.sum_error / (state.count + eps)


class FDE:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.state = fde_init()

    def update(self, pred_xy: jnp.ndarray, gt_xy: jnp.ndarray, valid: jnp.ndarray | None = None):
        self.state = fde_update(self.state, pred_xy, gt_xy, valid)

    def compute(self) -> jnp.ndarray:
        return fde_compute(self.state, self.eps)