import jax.numpy as jnp

from src.metrics import ade, fde


def test_ade_uses_all_valid_timesteps():
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

    agents_coeffs = jnp.ones((pred_xy.shape[0],), dtype=jnp.float32)
    assert jnp.isclose(
        ade(pred_xy, gt_xy, agents_coeffs, gt_mask),
        (1.0 + 2.0 + 4.0) / 3.0,
    )


def test_fde_uses_last_valid_timestep_per_agent():
    pred_xy = jnp.array(
        [
            [[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0], [0.0, 5.0]],
        ]
    )
    gt_xy = jnp.zeros_like(pred_xy)
    gt_mask = jnp.array([[True, True, False], [False, True, True]])

    agents_coeffs = jnp.ones((pred_xy.shape[0],), dtype=jnp.float32)
    assert jnp.isclose(
        fde(pred_xy, gt_xy, agents_coeffs, gt_mask),
        (3.0 + 5.0) / 2.0,
    )
