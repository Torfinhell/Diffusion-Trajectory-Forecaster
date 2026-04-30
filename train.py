from pathlib import Path

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
import jax.profiler
from hydra.utils import instantiate
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.models.base_model import Basetreainer
from src.utils import (
    JaxProfilerCallback,
    load_best_checkpoint,
    log_run_metadata,
    process_hparams,
)


@hydra.main(version_base=None, config_name="ddpm_attn", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=False)
    logger = instantiate(hparams.logger)
    profiler = None
    if cfg.trainer.enable_profiler:
        profiler = PyTorchProfiler(
            dirpath=f"./clearml/{logger.name}/profiler_logs", filename="perf_trace"
        )
    log_run_metadata(logger, hparams)
    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    callbacks = [RichProgressBar(leave=True)]
    if cfg.trainer.enable_jax_profiler:
        jax_profiler_dir = cfg.trainer.get("jax_profiler_dir")
        if jax_profiler_dir is None:
            jax_profiler_dir = f"./clearml/{logger.name}/jax_profiler"
        callbacks.append(
            JaxProfilerCallback(
                log_dir=jax_profiler_dir,
                start_step=cfg.trainer.get("jax_profiler_start_step", 2),
                num_steps=cfg.trainer.get("jax_profiler_num_steps", 3),
            )
        )
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
    diff_model = Basetreainer(
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
        prediction_target=hparams.prediction_target,
        t0=hparams.t0,
        t1=hparams.t1,
        dt0=hparams.dt0,
    )
    trainer.fit(diff_model, dm)
    if load_best_checkpoint(diff_model):
        trainer.test(diff_model, dm)
    else:
        print("Best checkpoint was not found. Skipping test run.")


if __name__ == "__main__":
    main()
