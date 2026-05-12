from typing import Callable, Iterable, Mapping

import jax.numpy as jnp

from .ade import ade
from .fde import fde

_METRIC_FNS: Mapping[str, Callable[..., jnp.ndarray]] = {
    "ade": ade,
    "fde": fde,
}


def _canonical_metric_name(name: str) -> str:
    return str(name).strip().lower()


class MetricFnCollection:
    """Hydra-instantiable metric collection.

    Stores already-jitted metric functions (`ade`, `fde`) and calls them.
    Can be invoked with explicit args or with `metrics(**batch)`; extra batch
    keys are ignored.
    """

    def __init__(self, metrics: Iterable[str], eps: float = 1e-8):
        self.eps = float(eps)
        names = []
        fns: list[Callable[..., jnp.ndarray]] = []
        for raw_name in metrics:
            name = _canonical_metric_name(raw_name)
            if name not in _METRIC_FNS:
                raise ValueError(
                    f"Unknown metric '{raw_name}'. Available: {sorted(_METRIC_FNS.keys())}"
                )
            names.append(name)
            fns.append(_METRIC_FNS[name])

        self.metric_names = tuple(names)
        self.metric_fns = tuple(fns)

    @property
    def names(self) -> tuple[str, ...]:
        return self.metric_names

    def __len__(self) -> int:
        return len(self.metric_names)

    def __call__(
        self,
        pred_xy: jnp.ndarray,
        gt_xy: jnp.ndarray,
        future_valid: jnp.ndarray,
        agents_valid: jnp.ndarray | None = None,
        agents_coeffs: jnp.ndarray | None = None,
        **_,
    ) -> dict[str, jnp.ndarray]:
        if agents_valid is None:
            if agents_coeffs is None:
                raise ValueError(
                    "MetricFnCollection requires `agents_valid` or `agents_coeffs`."
                )
            agents_valid = jnp.asarray(agents_coeffs) > 0

        res: dict[str, jnp.ndarray] = {}
        for name, fn in zip(self.metric_names, self.metric_fns):
            res[name.upper()] = fn(
                pred_xy=pred_xy,
                gt_xy=gt_xy,
                agents_valid=agents_valid,
                future_valid=future_valid,
                eps=self.eps,
            )
        return res
