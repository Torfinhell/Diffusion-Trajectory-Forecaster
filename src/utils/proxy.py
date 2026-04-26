import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def proxy_t_values(model):
    step_stride = max(1, int(model.proxy_val_cfg.get("step_stride", 5)))
    include_last = bool(model.proxy_val_cfg.get("include_last", True))
    t0 = float(getattr(model, "t0", 1e-3))
    t1 = float(model.t1)
    dt0 = abs(float(getattr(model, "dt0", 0.01)))
    num_steps = max(1, int(np.ceil((t1 - t0) / dt0)))
    ts = np.linspace(t0, t1, num_steps + 1, dtype=np.float32)[1:]
    selected = ts[step_stride - 1 :: step_stride]
    if include_last and (len(selected) == 0 or selected[-1] != ts[-1]):
        selected = np.concatenate([selected, ts[-1:]])
    return jnp.asarray(selected, dtype=jnp.float32)


def compute_proxy_batch_loss(model, batch, key):
    t_values = proxy_t_values(model)
    keys = jr.split(key, int(t_values.shape[0]))

    def loss_at_t(t, loss_key):
        return model.batch_loss_fn_fixed_t(
            model.model,
            model.int_beta,
            batch,
            t,
            loss_key,
        )

    losses = jax.vmap(loss_at_t)(t_values, keys)
    return jnp.mean(losses)
