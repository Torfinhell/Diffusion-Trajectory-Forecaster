import jax.numpy as jnp


def current_learning_rate_value(model) -> float:
    learning_rate = model.learning_rate_schedule
    if callable(learning_rate):
        return float(jnp.asarray(learning_rate(int(model.global_step_))))
    return float(learning_rate)


def log_training_step_stats(
    model,
    grad_norm,
    update_norm,
    param_norm,
    train_stats,
) -> None:
    if grad_norm is not None:
        model.log(
            "train/grad_norm",
            float(jnp.asarray(grad_norm)),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
    if update_norm is not None:
        model.log(
            "train/update_norm",
            float(jnp.asarray(update_norm)),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
    if param_norm is not None:
        model.log(
            "train/param_norm",
            float(jnp.asarray(param_norm)),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
    if train_stats is not None:
        if "target_abs_mean" in train_stats:
            model.log(
                "train/target_abs_mean",
                float(jnp.asarray(train_stats["target_abs_mean"])),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        if "pred_abs_mean" in train_stats:
            model.log(
                "train/pred_abs_mean",
                float(jnp.asarray(train_stats["pred_abs_mean"])),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        if "valid_ratio" in train_stats:
            model.log(
                "train/valid_ratio",
                float(jnp.asarray(train_stats["valid_ratio"])),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
    model.log(
        "train/lr",
        current_learning_rate_value(model),
        prog_bar=False,
        on_step=True,
        on_epoch=False,
    )
