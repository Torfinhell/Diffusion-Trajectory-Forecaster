import jax.numpy as jnp
import jax.random as jr

from mocked_model import OracleDiffusionModel


def test_oracle_diffusion_model_recovers_exact_training_target():
    gt_xy = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [0.25, -0.75]],
        ],
        dtype=jnp.float32,
    )
    model = OracleDiffusionModel(gt_xy=gt_xy, int_beta=lambda t: t)
    applied = model.apply_noise(t=0.7, key=jr.PRNGKey(0))

    pred = model(applied.t, applied.noisy_xy, cond=None)
    expected = -applied.noise / applied.std

    assert jnp.allclose(pred, expected)


def test_oracle_diffusion_model_infers_forward_noise():
    gt_xy = jnp.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=jnp.float32)
    model = OracleDiffusionModel(gt_xy=gt_xy, int_beta=lambda t: t)
    applied = model.apply_noise(t=1.3, key=jr.PRNGKey(1))

    inferred_noise = model.infer_noise(applied.t, applied.noisy_xy)

    assert jnp.allclose(inferred_noise, applied.noise)
