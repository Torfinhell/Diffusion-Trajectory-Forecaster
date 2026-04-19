from dataclasses import dataclass

import jax.numpy as jnp

from .base import BaseMetric, temporal_valid_mask


@dataclass
class FDEState:
    sum_error: jnp.ndarray
    count: jnp.ndarray


class Static_metrics(BaseMetric):
    def __init__(self, *names, **kwargs):
        super().__init__(**kwargs)
        self.reset()
        self.names = names

    def reset(self):
        self.sum_met = {name: 0 for name in self.names}
        self.count_met = {name: 0 for name in self.names}

    def compute(self):
        dict_compute = {}
        for name in self.names:
            dict_compute.update({name: self.sum_met[name] / self.count_met[name]})
        return dict_compute

    def update(self, name, value) -> None:
        self.sum_met[name] += value
        self.count_met[name] += 1
