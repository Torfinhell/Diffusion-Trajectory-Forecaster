import jax.numpy as jnp

def stack_metric(metrics_t, name):
    """metrics_t: list[dict[str, MetricResult]] -> (vals[T,...], valid[T,...])"""
    vals = jnp.stack([m[name].value for m in metrics_t], axis=0)
    valid = jnp.stack([m[name].valid for m in metrics_t], axis=0)
    return vals, valid

def mean_over_valid(vals, valid):
    vals = vals.astype(jnp.float32)
    valid = valid.astype(jnp.float32)
    denom = jnp.maximum(1.0, jnp.sum(valid))
    return jnp.sum(vals * valid) / denom

def any_over_time(vals, valid):
    vals = vals.astype(jnp.float32)
    return jnp.any((vals > 0) & valid)

def rate_over_valid(vals, valid):
    vals = vals.astype(jnp.float32)
    denom = jnp.maximum(1.0, jnp.sum(valid))
    return jnp.sum(((vals > 0) & valid).astype(jnp.float32)) / denom

def summarize_episode_metrics(metrics_t, ego_mask=None):
    """
    ego_mask: shape [num_objects] bool; if None -> all objects.
    returns python-friendly scalars.
    """
    out = {}

    for name, mode in [
        ("overlap", "any"),
        ("offroad", "any"),
        ("wrong_way", "rate"),
        ("log_divergence", "mean"),
    ]:
        if name not in metrics_t[0]:
            continue

        vals, valid = stack_metric(metrics_t, name)  # [T, N] 

        if ego_mask is not None:
            vals = vals[:, ego_mask]
            valid = valid[:, ego_mask]

        if mode == "mean":
            out[name] = float(mean_over_valid(vals, valid))
        elif mode == "any":
            out[name] = bool(any_over_time(vals, valid))
        elif mode == "rate":
            out[name] = float(rate_over_valid(vals, valid))

    return out

def average_episode_summaries(episode_summaries):
    """
    episode_summaries: list[dict[str, scalar]]
    -> dict averaged across episodes (booleans become rate)
    """
    keys = set().union(*episode_summaries)
    out = {}
    for k in keys:
        vals = [s[k] for s in episode_summaries if k in s]
        # booleans -> mean rate
        if isinstance(vals[0], (bool, jnp.bool_)):
            out[k] = float(sum(bool(v) for v in vals) / len(vals))
        else:
            out[k] = float(sum(vals) / len(vals))
    return out
