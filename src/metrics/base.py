from abc import abstractmethod

import jax.numpy as jnp


def traj_to_xy_valid(traj):
    # traj.x, traj.y
    xy = jnp.stack([traj.x, traj.y], axis=-1)  # (..., T, 2)
    valid = getattr(traj, "valid", None)
    return xy, valid


class BaseMetric:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, pred_xy, gt_xy, valid):
        pass

    @abstractmethod
    def compute(self):
        pass
