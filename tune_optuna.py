import json
from copy import deepcopy
from pathlib import Path

import hydra
import optuna
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.trainers import BaseTrainer
from src.utils import (
    build_training_modules,
    process_hparams,
    resolve_epoch_len,
    resolve_scheduler_decay_steps,
    split_trainer_config,
)


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def apply_scaled_model_params(cfg, trial: optuna.Trial, tune_cfg) -> None:
    width = trial.suggest_categorical("width", list(tune_cfg.search.widths))
    attn_mlp_ratio = trial.suggest_categorical(
        "attn_mlp_ratio", list(tune_cfg.search.attn_mlp_ratios)
    )
    num_sa_mlp = trial.suggest_int(
        "num_sa_mlp",
        int(tune_cfg.search.num_sa_mlp.min),
        int(tune_cfg.search.num_sa_mlp.max),
    )
    num_camlp = trial.suggest_int(
        "num_camlp",
        int(tune_cfg.search.num_camlp.min),
        int(tune_cfg.search.num_camlp.max),
    )

    attn_mlp_dim = _round_to_multiple(
        width * attn_mlp_ratio, int(tune_cfg.search.mlp_multiple)
    )

    cfg.model.se_args.out_dim = width
    cfg.model.camlp_args.kv_dim = width
    cfg.model.samlp_args.mlp_dim = attn_mlp_dim
    cfg.model.camlp_args.mlp_dim = attn_mlp_dim
    cfg.model.num_sa_mlp = num_sa_mlp
    cfg.model.num_camlp = num_camlp

    trial.set_user_attr("attn_mlp_dim", attn_mlp_dim)


def build_hparams(tune_cfg, trial: optuna.Trial):
    cfg = compose(config_name=tune_cfg.base_config_name)

    OmegaConf.set_struct(cfg, False)
    cfg = deepcopy(cfg)

    apply_scaled_model_params(cfg, trial, tune_cfg)

    batch_size = trial.suggest_categorical(
        "batch_size", list(tune_cfg.search.batch_sizes)
    )
    lr0 = trial.suggest_float(
        "lr0",
        float(tune_cfg.search.lr0.min),
        float(tune_cfg.search.lr0.max),
        log=True,
    )
    lrf = trial.suggest_float(
        "lrf",
        float(tune_cfg.search.lrf.min),
        float(tune_cfg.search.lrf.max),
        log=True,
    )

    cfg.dataloaders.train.batch_size = batch_size
    cfg.dataloaders.val.batch_size = batch_size
    cfg.dataloaders.test.batch_size = batch_size
    cfg.scheduler.init_value = lr0
    cfg.scheduler.alpha = lrf

    cfg.trainer.num_epochs = int(tune_cfg.trainer.num_epochs)
    cfg.trainer.train_epoch_len = tune_cfg.trainer.train_epoch_len
    cfg.trainer.val_epoch_len = tune_cfg.trainer.val_epoch_len
    cfg.trainer.enable_profiler = False
    cfg.trainer.enable_jax_profiler = False
    cfg.trainer.logging = "disable"
    cfg.visual.enable_visualization = False
    cfg.visual.enable_train_visualization = False

    return process_hparams(cfg, print_hparams=False)


def objective(tune_cfg, trial: optuna.Trial) -> float:
    hparams = build_hparams(tune_cfg, trial)
    dm = DiffusionTrackerDataModule(hparams.dataset.data, hparams.dataloaders)
    dm.setup("fit")
    resolve_scheduler_decay_steps(hparams, dm)
    train_epoch_len = resolve_epoch_len(
        hparams.trainer.train_epoch_len, len(dm.train_dataloader())
    )
    val_epoch_len = resolve_epoch_len(
        hparams.trainer.val_epoch_len, len(dm.val_dataloader())
    )
    val_check_interval = train_epoch_len * int(hparams.trainer.check_val_every_n_epoch)

    _, module_trainer_cfg = split_trainer_config(hparams.trainer)
    module_trainer_cfg = {**module_trainer_cfg, "loss_returns_stats": True}
    train_mode = hparams.trainer.get("train_mode", "train")
    modules = build_training_modules(hparams, train_mode)
    diff_model = BaseTrainer(
        cfg_metrics=hparams.metrics,
        vis_cfg=hparams.visual,
        **modules,
        **module_trainer_cfg,
    )

    trainer = Trainer(
        accelerator=tune_cfg.runtime.accelerator,
        devices=1,
        max_epochs=hparams.trainer.num_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(tune_cfg.runtime.show_progress),
        limit_train_batches=train_epoch_len,
        limit_val_batches=val_epoch_len,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )

    try:
        trainer.fit(diff_model, dm)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise optuna.TrialPruned(f"OOM: {exc}") from exc
        raise

    metric_key = str(tune_cfg.objective.metric)
    if metric_key not in trainer.callback_metrics:
        raise RuntimeError(
            f"Expected metric '{metric_key}' was not logged. "
            f"Available metrics: {sorted(trainer.callback_metrics.keys())}"
        )
    metric_value = float(trainer.callback_metrics[metric_key])
    trial.set_user_attr(metric_key, metric_value)
    return metric_value


def save_study_results(study: optuna.Study) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_trial_payload = {
        "value": study.best_value,
        "params": dict(study.best_trial.params),
        "user_attrs": dict(study.best_trial.user_attrs),
    }
    (output_dir / "best_trial.json").write_text(
        json.dumps(best_trial_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "trials.csv", index=False)


@hydra.main(version_base=None, config_path="src/configs", config_name="tune_optuna")
def main(cfg) -> None:
    study = optuna.create_study(
        study_name=cfg.study.name,
        storage=cfg.study.storage,
        direction=cfg.study.direction,
        load_if_exists=bool(cfg.study.load_if_exists),
    )
    study.optimize(
        lambda trial: objective(cfg, trial),
        n_trials=int(cfg.study.n_trials),
        timeout=cfg.study.timeout,
    )
    save_study_results(study)

    print("Best trial:")
    print(f"  value: {study.best_value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
