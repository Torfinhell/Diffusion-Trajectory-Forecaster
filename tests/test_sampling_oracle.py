import jax.numpy as jnp
import jax.random as jr

from mocked_model import OracleDiffusionModel
from src.metrics import AdeMetric, FdeMetric
from src.models.base_model import BaseDiffusionModel


class _SamplingHarness:
    def __init__(self, dt0=0.1):
        self.model = None
        self.int_beta = lambda t: t
        self.dt0 = dt0
        self.t1 = 1.0
        self.prediction_target = "score"
        self.sample_key = jr.PRNGKey(0)
        self.oracle_cfg = {"sampling_mode": "ode", "sampling_t0": 1e-3}

    def _make_oracle_model(self, gt_xy):
        return OracleDiffusionModel(gt_xy=gt_xy, int_beta=self.int_beta)

    def _oracle_sampling_mode(self):
        return self.oracle_cfg["sampling_mode"]

    def _sampling_t0(self):
        return self.oracle_cfg["sampling_t0"]


_SamplingHarness._alpha_sigma = staticmethod(BaseDiffusionModel._alpha_sigma)
_SamplingHarness.sample_one_sol = BaseDiffusionModel.sample_one_sol
_SamplingHarness.sample_multiple_sol = BaseDiffusionModel.sample_multiple_sol


def _compute_metrics(pred_xy, gt_xy):
    ade = AdeMetric(name="ADE")
    fde = FdeMetric(name="FDE")
    mask = jnp.ones(gt_xy.shape[:-1], dtype=bool)
    ade.update(pred_xy, gt_xy, mask)
    fde.update(pred_xy, gt_xy, mask)
    return float(ade.compute()), float(fde.compute())


def _closed_form_pf_solution(gt_xy, noise, t):
    alpha = jnp.exp(-0.5 * t)
    sigma = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-t), 1e-5))
    return alpha * gt_xy + sigma * noise


def test_oracle_ode_sampling_recovers_gt_from_known_xt1():
    harness = _SamplingHarness(dt0=1e-3)
    gt_xy = jnp.array(
        [[[1.0, -2.0], [0.5, 0.0], [2.0, 1.5]]],
        dtype=jnp.float32,
    )
    oracle = OracleDiffusionModel(gt_xy=gt_xy, int_beta=harness.int_beta)
    applied = oracle.apply_noise(t=harness.t1, key=jr.PRNGKey(7))

    pred_xy, pred_paths = harness.sample_multiple_sol(
        context=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        num_solutions=1,
        predict_shape=gt_xy.shape,
        oracle_gt_xy=gt_xy,
        y1_override=applied.noisy_xy,
    )

    expected_final = _closed_form_pf_solution(
        gt_xy, applied.noise, harness.oracle_cfg["sampling_t0"]
    )
    ade, fde = _compute_metrics(pred_xy, expected_final)

    assert pred_paths.shape == (1, *gt_xy.shape)
    assert ade < 1e-3
    assert fde < 1e-3


def test_oracle_ode_sampling_improves_with_smaller_dt0():
    gt_xy = jnp.array(
        [[[1.0, 2.0], [3.0, -1.0], [0.25, 0.5]]],
        dtype=jnp.float32,
    )
    oracle = OracleDiffusionModel(gt_xy=gt_xy, int_beta=lambda t: t)
    applied = oracle.apply_noise(t=1.0, key=jr.PRNGKey(11))

    coarse = _SamplingHarness(dt0=0.1)
    fine = _SamplingHarness(dt0=0.01)

    coarse_pred, _ = coarse.sample_multiple_sol(
        context=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        num_solutions=1,
        predict_shape=gt_xy.shape,
        oracle_gt_xy=gt_xy,
        y1_override=applied.noisy_xy,
    )
    fine_pred, _ = fine.sample_multiple_sol(
        context=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        num_solutions=1,
        predict_shape=gt_xy.shape,
        oracle_gt_xy=gt_xy,
        y1_override=applied.noisy_xy,
    )

    coarse_ade, coarse_fde = _compute_metrics(coarse_pred, gt_xy)
    fine_ade, fine_fde = _compute_metrics(fine_pred, gt_xy)

    assert fine_ade <= coarse_ade + 1e-6


def test_oracle_ode_sampling_matches_closed_form_path():
    harness = _SamplingHarness(dt0=1e-3)
    gt_xy = jnp.array(
        [[[1.0, -2.0], [0.5, 0.0], [2.0, 1.5]]],
        dtype=jnp.float32,
    )
    oracle = OracleDiffusionModel(gt_xy=gt_xy, int_beta=harness.int_beta)
    applied = oracle.apply_noise(t=harness.t1, key=jr.PRNGKey(5))

    pred_path = harness.sample_one_sol(
        model=None,
        int_beta=harness.int_beta,
        data_shape=gt_xy.shape,
        dt0=harness.dt0,
        t1=harness.t1,
        context=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        save_full=True,
        oracle_gt_xy=gt_xy,
        y1_override=applied.noisy_xy,
        key=jr.PRNGKey(0),
    )
    ts = jnp.linspace(
        harness.t1, harness.oracle_cfg["sampling_t0"], pred_path.shape[0]
    )
    expected_path = _closed_form_pf_solution(
        gt_xy[None, ...], applied.noise[None, ...], ts[:, None, None, None]
    )

    path_err = jnp.max(jnp.abs(pred_path - expected_path))
    assert path_err < 1e-3
