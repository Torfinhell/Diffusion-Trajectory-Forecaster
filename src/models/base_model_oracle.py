import jax.numpy as jnp
import jax.random as jr

try:
    from mocked_model import OracleDiffusionModel
except ImportError:
    OracleDiffusionModel = None


def oracle_enabled(model, key):
    return bool(model.oracle_cfg is not None and model.oracle_cfg.get(key, False))


def make_oracle_model(model, gt_xy):
    if OracleDiffusionModel is None:
        raise RuntimeError(
            "Oracle mode requested but mocked_model.py is not available."
        )
    return OracleDiffusionModel(gt_xy=gt_xy, int_beta=model.int_beta)


def oracle_sampling_mode(model):
    if model.oracle_cfg is None:
        return "exact"
    return model.oracle_cfg.get("sampling_mode", "exact")


def sampling_t0(model):
    if hasattr(model, "t0"):
        return float(model.t0)
    if model.oracle_cfg is None:
        return 1e-3
    return float(model.oracle_cfg.get("sampling_t0", 1e-3))


def is_oracle_model(instance):
    return OracleDiffusionModel is not None and isinstance(
        instance, OracleDiffusionModel
    )


def compute_batch_loss(model, batch, key, use_oracle=False):
    if not use_oracle:
        return model.batch_loss_fn(
            model.model,
            model.weight,
            model.int_beta,
            model.prediction_target,
            batch,
            model.t1,
            key,
        )

    batch_size = batch["gt_xy"].shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=model.t1 / batch_size)
    t = t + (model.t1 / batch_size) * jnp.arange(batch_size)

    losses = []
    for sample_idx in range(batch_size):
        sample_batch = {
            "gt_xy": batch["gt_xy"][sample_idx],
            "gt_xy_mask": batch["gt_xy_mask"][sample_idx],
            "context": batch["context"][sample_idx],
        }
        oracle_model = make_oracle_model(model, sample_batch["gt_xy"])
        losses.append(
            model.single_loss_fn(
                oracle_model,
                model.weight,
                model.int_beta,
                model.prediction_target,
                sample_batch,
                t[sample_idx],
                losskey[sample_idx],
            )
        )
    return jnp.mean(jnp.stack(losses))
