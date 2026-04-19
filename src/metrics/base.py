from abc import abstractmethod

import jax.numpy as jnp


def traj_to_xy_valid(traj):
    # traj.x, traj.y
    xy = jnp.stack([traj.x, traj.y], axis=-1)  # (..., T, 2)
    valid = getattr(traj, "valid", None)
    return xy, valid


def temporal_valid_mask(gt_xy, valid):
    """Normalizes valid masks to shape (..., T)."""
    if valid is None:
        return jnp.ones(gt_xy.shape[:-1], dtype=jnp.bool_)

    valid = jnp.asarray(valid)
    if valid.shape == gt_xy.shape[:-1]:
        return valid.astype(jnp.bool_)
    if valid.shape == gt_xy.shape:
        return jnp.all(valid.astype(jnp.bool_), axis=-1)

    raise ValueError(
        "Expected valid mask with shape matching gt_xy[..., :-1] or gt_xy, "
        f"got {valid.shape} for gt_xy {gt_xy.shape}."
    )


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


class MetricCollection:
    def __init__(self, metrics: list[BaseMetric]):
        self.metrics = metrics

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, *args):
        for metric in self.metrics:
            metric.update(*args)

    def compute(self):
        res = {}
        for metric in self.metrics:
            res[metric.name] = metric.compute()
        return res

    def __len__(self):
        return len(self.metrics)
