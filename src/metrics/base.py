import jax.numpy as jnp
from ade import ADE
from fde import FDE

def traj_to_xy_valid(traj):
    # traj.x, traj.y
    xy = jnp.stack([traj.x, traj.y], axis=-1)   # (..., T, 2)
    valid = getattr(traj, "valid", None)        
    return xy, valid

def build_metrics(names):
    reg = {}
    for n in names:
        if n == "ade":
            reg["ADE"] = ADE()
        elif n == "fde":
            reg["FDE"] = FDE()
        else:
            raise ValueError(f"Unknown metric: {n}")
    return reg

class MetricsManager:
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def update(self, pred_xy, gt_xy, valid):
        for m in self.metrics.values():
            m.update(pred_xy, gt_xy, valid)

    def compute(self):
        return {name: m.compute() for name, m in self.metrics.items()}