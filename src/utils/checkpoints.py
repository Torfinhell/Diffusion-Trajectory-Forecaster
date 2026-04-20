from pathlib import Path

import equinox as eqx
import jax.numpy as jnp


def sanitize_checkpoint_name_component(value, fallback="unknown"):
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def build_checkpoint_run_directory(root: Path, logger) -> Path:
    version = sanitize_checkpoint_name_component(getattr(logger, "version", None), "local")
    return root / version


def checkpoint_run_dir(model) -> Path:
    logger = getattr(model, "logger", None)
    return build_checkpoint_run_directory(model.CHECKPOINT_ROOT, logger)


def best_checkpoint_path(model) -> Path:
    return checkpoint_run_dir(model) / "best.eqx"


def maybe_save_best_checkpoint(model, metrics) -> None:
    metric_value = metrics.get(model.best_checkpoint_metric)
    if metric_value is None:
        return

    score = float(jnp.asarray(metric_value))
    improved = (
        score < model.best_checkpoint_score
        if model.best_checkpoint_mode == "min"
        else score > model.best_checkpoint_score
    )
    if not improved:
        return

    checkpoint_path = best_checkpoint_path(model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_path, model.model)
    model.best_checkpoint_score = score
    model.best_checkpoint_epoch = int(model.current_epoch)
    print(
        f"Saved best checkpoint to {checkpoint_path} "
        f"({model.best_checkpoint_metric}={score:.6f})"
    )


def log_model_artifact(model) -> None:
    logger = getattr(model, "logger", None)
    if logger is None:
        return

    checkpoint_path = best_checkpoint_path(model)
    if not checkpoint_path.exists():
        return

    if hasattr(logger, "upload_artifact"):
        logger.upload_artifact(
            name="best_checkpoint",
            path=checkpoint_path,
            metadata={
                "epoch": model.best_checkpoint_epoch,
                "global_step": int(model.global_step_),
                "monitor_metric": model.best_checkpoint_metric,
                "monitor_mode": model.best_checkpoint_mode,
                "monitor_score": model.best_checkpoint_score,
            },
        )


def load_best_checkpoint(model) -> bool:
    checkpoint_path = best_checkpoint_path(model)
    if not checkpoint_path.exists():
        return False
    model.model = eqx.tree_deserialise_leaves(checkpoint_path, model.model)
    print(f"Loaded best checkpoint from {checkpoint_path}")
    return True
