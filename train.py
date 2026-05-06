import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
import hydra
from hydra.utils import instantiate
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.utils import (
    JaxProfilerCallback,
    log_run_metadata,
    process_hparams,
    resolve_scheduler_decay_steps,
)
from trainers import BaseProfilerDebug, BaseTrainer, BaseTrainerDebug


@hydra.main(version_base=None, config_name="ddpm_attn", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=False)
    logger = instantiate(hparams.logger) if getattr(hparams, "logger", None) else None
    if logger:
        log_run_metadata(logger, hparams)

    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    dm.setup("fit")
    resolve_scheduler_decay_steps(hparams, dm)

    debug_type = cfg.trainer.get("debug_type", None)
    trainer_mapping = {
        None: BaseTrainer,
        "debug": BaseTrainerDebug,
        "profiler": BaseProfilerDebug,
    }

    if debug_type not in trainer_mapping:
        raise NotImplementedError(f"Debugging of type {debug_type} is not implemented")

    diff_trainer_kwargs = dict(
        seed=hparams.trainer.seed,
        load_best_checkpoint=hparams.trainer.load_best_checkpoint,
        cfg_metrics=hparams.metrics,
        vis_cfg=hparams.visual,
        model=hparams.model,
        loss=hparams.loss,
        optimizer=hparams.optimizer,
        scheduler=hparams.scheduler,
        diffusion_sampler=hparams.diffusion_sampler,
        grad_clip=hparams.trainer.gradient_clip_val,
        trainer_cfg=hparams.trainer,
    )

    diff_trainer = trainer_mapping[debug_type](**diff_trainer_kwargs)

    callbacks = [RichProgressBar(leave=True)]
    profiler = None

    if debug_type == "profiler":
        log_name = logger.name if logger else "default_run"
        jax_profiler_dir = f"./clearml/{log_name}/jax_profiler"
        profiler = JaxProfilerCallback(
            log_dir=jax_profiler_dir,
            start_step=cfg.trainer.get("jax_profiler_start_step", 2),
            num_steps=cfg.trainer.get("jax_profiler_num_steps", 3),
        )
        callbacks.append(profiler)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=hparams.trainer.num_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=hparams.trainer.log_every_n_steps,
        enable_progress_bar=True,
        limit_train_batches=hparams.trainer.train_epoch_len,
        limit_val_batches=hparams.trainer.val_epoch_len,
        check_val_every_n_epoch=hparams.trainer.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=cfg.trainer.generate_every_epoch,
        profiler=profiler,
    )

    trainer.fit(diff_trainer, dm)


if __name__ == "__main__":
    main()
