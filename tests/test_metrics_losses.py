import jax.numpy as jnp

from src.metrics import AdeMetric, FdeMetric


def test_ade_uses_all_valid_timesteps():
    metric = AdeMetric(name="ADE")
    pred_xy = jnp.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [100.0, 0.0]],
            [[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]],
        ]
    )
    gt_xy = jnp.zeros_like(pred_xy)
    gt_mask = jnp.array(
        [
            [True, True, False],
            [False, True, False],
        ]
    )

    metric.update(pred_xy, gt_xy, gt_mask)

    assert jnp.isclose(metric.compute(), (1.0 + 2.0 + 4.0) / 3.0)


def test_fde_uses_last_valid_timestep_per_agent():
    metric = FdeMetric(name="FDE")
    pred_xy = jnp.array(
        [
            [[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0], [0.0, 5.0]],
        ]
    )
    gt_xy = jnp.zeros_like(pred_xy)
    gt_mask = jnp.array(
        [
            [[True, True], [True, True], [False, False]],
            [[False, False], [True, True], [True, True]],
        ]
    )

    metric.update(pred_xy, gt_xy, gt_mask)

    assert jnp.isclose(metric.compute(), (3.0 + 5.0) / 2.0)
