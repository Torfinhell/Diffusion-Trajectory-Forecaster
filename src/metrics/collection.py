from typing import Callable, Iterable, Mapping

import jax.numpy as jnp

from .ade import ade
from .fde import fde
from .object_types import DEFAULT_TYPE_LABELS

_METRIC_FNS: Mapping[str, Callable[..., jnp.ndarray]] = {
    "ade": ade,
    "fde": fde,
}


def _type_label(type_id: int, labels: Mapping[int, str] | None) -> str:
    if labels is not None and int(type_id) in labels:
        return str(labels[int(type_id)])
    return DEFAULT_TYPE_LABELS.get(int(type_id), str(int(type_id)))


def _agent_weights_for_type(
    agents_coeffs: jnp.ndarray,
    agents_types: jnp.ndarray,
    type_id: int,
) -> jnp.ndarray:
    agents_coeffs = jnp.asarray(agents_coeffs, dtype=jnp.float32)
    agents_types = jnp.asarray(agents_types)
    return agents_coeffs * (agents_types == int(type_id)).astype(jnp.float32)


class MetricFnCollection:
    """Hydra-instantiable metric collection with optional per-object-type breakdown."""

    def __init__(
        self,
        metrics: Iterable[str],
        eps: float = 1e-8,
        object_type_ids: Iterable[int] | None = None,
        object_type_labels: Mapping[int, str] | None = None,
    ):
        self.eps = float(eps)
        self.object_type_ids = tuple(int(i) for i in (object_type_ids or ()))
        self.object_type_labels = (
            {int(k): str(v) for k, v in dict(object_type_labels).items()}
            if object_type_labels
            else None
        )

        names: list[str] = []
        fns: list[Callable[..., jnp.ndarray]] = []
        for raw_name in metrics:
            name = str(raw_name).strip().lower()
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

    def _run_metrics(
        self,
        pred_xy: jnp.ndarray,
        gt_xy: jnp.ndarray,
        agents_coeffs: jnp.ndarray,
        future_valid: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        res: dict[str, jnp.ndarray] = {}
        for name, fn in zip(self.metric_names, self.metric_fns):
            res[name.upper()] = fn(
                pred_xy=pred_xy,
                gt_xy=gt_xy,
                agents_coeffs=agents_coeffs,
                future_valid=future_valid,
                eps=self.eps,
            )
        return res

    def __call__(
        self,
        pred_xy: jnp.ndarray,
        gt_xy: jnp.ndarray,
        agents_coeffs: jnp.ndarray,
        future_valid: jnp.ndarray,
        agents_types: jnp.ndarray | None = None,
        **_,
    ) -> dict[str, jnp.ndarray]:
        res = self._run_metrics(pred_xy, gt_xy, agents_coeffs, future_valid)

        if not self.object_type_ids or agents_types is None:
            return res

        for type_id in self.object_type_ids:
            label = _type_label(type_id, self.object_type_labels)
            type_coeffs = _agent_weights_for_type(agents_coeffs, agents_types, type_id)
            for name, fn in zip(self.metric_names, self.metric_fns):
                key = f"{name.upper()}_{label}"
                res[key] = fn(
                    pred_xy=pred_xy,
                    gt_xy=gt_xy,
                    agents_coeffs=type_coeffs,
                    future_valid=future_valid,
                    eps=self.eps,
                )
        return res
