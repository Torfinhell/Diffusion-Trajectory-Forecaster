import gc
from copy import deepcopy
from typing import Any

import hydra
import optuna
from hydra.utils import instantiate
from optuna.exceptions import TrialPruned
from pytorch_lightning import Callback
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.utils import process_hparams


class OptunaPruningCallback(Callback):
    def __init__(self, trial: optuna.Trial, metric_name: str, direction: str):
        super().__init__()
        self.trial = trial
        self.metric_name = metric_name
        self.direction = str(direction).lower()
        self.best_value: float | None = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        metric = trainer.callback_metrics.get(self.metric_name)
        if metric is None:
            return

        value = float(metric)
        step = int(trainer.current_epoch)
        self.trial.report(value, step=step)

        if self.best_value is None:
            self.best_value = value
        elif self.direction == "minimize":
            self.best_value = min(self.best_value, value)
        else:
            self.best_value = max(self.best_value, value)

        if self.trial.should_prune():
            raise TrialPruned(
                f"Trial pruned at epoch {step} with {self.metric_name}={value:.6f}"
            )


def _set_by_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _suggest_value(trial: optuna.Trial, name: str, spec: Any):
    spec_type = str(spec["type"]).lower()

    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=float(spec["step"]) if spec.get("step") is not None else None,
        )
    if spec_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            log=bool(spec.get("log", False)),
            step=int(spec.get("step", 1)),
        )
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))

    raise ValueError(
        f"Unsupported Optuna parameter type '{spec_type}' for search space entry '{name}'."
    )


def _make_trainer(cfg: Any, trial: optuna.Trial) -> tuple[Trainer, OptunaPruningCallback]:
    metric_name = str(cfg.optuna.metric)
    direction = str(cfg.optuna.direction)
    pruning_cb = OptunaPruningCallback(trial, metric_name=metric_name, direction=direction)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=cfg.trainer.num_epochs,
        logger=False,
        callbacks=[pruning_cb],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=False,
        limit_train_batches=cfg.trainer.train_epoch_len,
        limit_val_batches=cfg.trainer.val_epoch_len,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=cfg.trainer.generate_every_epoch,
        profiler=None,
    )
    return trainer, pruning_cb


def _run_trial(base_cfg: Any, trial: optuna.Trial) -> float:
    cfg = deepcopy(base_cfg)

    for path, spec in cfg.optuna.search_space.items():
        _set_by_path(cfg, path, _suggest_value(trial, path, spec))
    for key, value in cfg.optuna.trainer_overrides.items():
        _set_by_path(cfg, f"trainer.{key}", value)
    if cfg.optuna.proxy_metric.enabled:
        cfg.trainer.proxy_val_loss = {
            "enabled": True,
            "step_stride": int(cfg.optuna.proxy_metric.step_stride),
            "include_last": bool(cfg.optuna.proxy_metric.include_last),
        }
        if bool(cfg.optuna.proxy_metric.disable_sampling_metrics):
            cfg.trainer.val_metric_every_n_epochs = int(cfg.trainer.num_epochs) + 1

    hparams = process_hparams(cfg, print_hparams=bool(cfg.trainer.print_hparams))
    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    model = instantiate(hparams.model, _recursive_=False)
    trainer, pruning_cb = _make_trainer(hparams, trial)

    try:
        trainer.fit(model, dm)
    finally:
        del trainer
        del model
        del dm
        if bool(cfg.optuna.gc_after_trial):
            gc.collect()

    if pruning_cb.best_value is None:
        raise RuntimeError(
            f"Optuna metric '{cfg.optuna.metric}' was never logged during the trial."
        )
    return pruning_cb.best_value


@hydra.main(version_base=None, config_name="ddpm_baseline", config_path="src/configs")
def main(cfg: Any) -> None:
    if not bool(cfg.optuna.enabled):
        raise RuntimeError(
            "Optuna tuning is disabled. Set `optuna.enabled=true` and define `optuna.search_space`."
        )
    if len(cfg.optuna.search_space) == 0:
        raise RuntimeError(
            "Optuna search space is empty. Fill `optuna.search_space` with dotted config paths."
        )

    if cfg.optuna.pruner.enabled:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(cfg.optuna.pruner.n_startup_trials),
            n_warmup_steps=int(cfg.optuna.pruner.n_warmup_steps),
            interval_steps=int(cfg.optuna.pruner.interval_steps),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction=str(cfg.optuna.direction),
        storage=cfg.optuna.storage,
        load_if_exists=bool(cfg.optuna.load_if_exists),
        sampler=optuna.samplers.TPESampler(seed=int(cfg.optuna.sampler.seed)),
        pruner=pruner,
    )
    study.optimize(
        lambda trial: _run_trial(cfg, trial),
        n_trials=int(cfg.optuna.n_trials),
        gc_after_trial=bool(cfg.optuna.gc_after_trial),
    )

    print("Best trial:")
    print(f"  value: {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
