from abc import abstractmethod

import jax.numpy as jnp

class BaseMetric:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, pred_xy, gt_xy, agents_coeffs):
        pass

    @abstractmethod
    def compute(self):
        pass


class MetricCollection:
    def __init__(self, metrics: list[BaseMetric]):
        self.metrics = metrics

    def add(self, metric):
        self.metrics.append(metric)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, pred_xy, gt_xy, agents_coeffs):
        for metric in self.metrics:
            metric.update(pred_xy, gt_xy, agents_coeffs)

    def compute(self):
        res = {}
        for metric in self.metrics:
            res[metric.name] = metric.compute()
        return res

    def __len__(self):
        return len(self.metrics)
