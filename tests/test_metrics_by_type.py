import jax.numpy as jnp

from src.metrics import MetricFnCollection


def test_per_type_metrics():
    pred_xy = jnp.array(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 4.0]],
            [[10.0, 0.0], [10.0, 0.0]],
        ]
    )
    gt_xy = jnp.zeros_like(pred_xy)
    future_valid = jnp.ones((3, 2, 1))
    agents_coeffs = jnp.ones((3,))
    agents_types = jnp.array([1, 2, 1], dtype=jnp.int32)

    metrics = MetricFnCollection(
        metrics=["ade"],
        object_type_ids=[1, 2],
        object_type_labels={1: "vehicle", 2: "pedestrian"},
    )
    res = metrics(
        pred_xy=pred_xy,
        gt_xy=gt_xy,
        agents_coeffs=agents_coeffs,
        future_valid=future_valid,
        agents_types=agents_types,
    )
    assert "ADE" in res
    assert jnp.isclose(res["ADE"], (1.0 + 2.0 + 4.0) / 3.0)
    assert jnp.isclose(res["ADE_vehicle"], (1.0 + 2.0) / 2.0)
    assert jnp.isclose(res["ADE_pedestrian"], 4.0)
