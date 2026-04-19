from pathlib import Path

import hydra
import jax.profiler
from hydra.utils import instantiate
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.trainer import Trainer

from src.data_module import DiffusionTrackerDataModule
from src.utils import log_run_metadata, process_hparams


class JaxProfilerCallback(Callback):
    def __init__(self, log_dir: str, start_step: int, num_steps: int):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.start_step = int(start_step)
        self.stop_step = self.start_step + int(num_steps)
        self._active = False
        self._completed = False
        self._seen_train_batches = 0

    def on_fit_start(self, trainer, pl_module):
        del trainer, pl_module
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.stop_step <= self.start_step:
            raise ValueError("JAX profiler num_steps must be a positive integer.")
        self._seen_train_batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, pl_module, batch, batch_idx
        if self._completed or self._active:
            return
        if self._seen_train_batches >= self.start_step:
            jax.profiler.start_trace(str(self.log_dir))
            self._active = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del trainer, pl_module, outputs, batch, batch_idx
        self._seen_train_batches += 1
        if self._active and self._seen_train_batches >= self.stop_step:
            jax.profiler.stop_trace()
            self._active = False
            self._completed = True

    def on_fit_end(self, trainer, pl_module):
        del trainer, pl_module
        if self._active:
            jax.profiler.stop_trace()
            self._active = False
            self._completed = True


class ClearMLFlushCallback(Callback):
    def _maybe_flush(self, trainer, force: bool = False):
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "maybe_flush_metrics"):
            return
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        logger.maybe_flush_metrics(epoch=epoch, force=force)

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        self._maybe_flush(trainer)

    def on_fit_end(self, trainer, pl_module):
        del pl_module
        self._maybe_flush(trainer, force=True)


@hydra.main(version_base=None, config_name="ddpm_1", config_path="src/configs")
def main(cfg) -> None:
    hparams = process_hparams(cfg, print_hparams=True)
    logger = instantiate(hparams.logger)
    profiler = None
    if cfg.trainer.enable_profiler:
        profiler = PyTorchProfiler(
            dirpath=f"./clearml/{logger.name}/profiler_logs", filename="perf_trace"
        )
    log_run_metadata(logger, hparams)
    dm = DiffusionTrackerDataModule(hparams.data, hparams.dataloaders)
    callbacks = [RichProgressBar(leave=True), ClearMLFlushCallback()]
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
    diff_model = instantiate(
        hparams.model,
        _recursive_=False,
    )
    trainer.fit(diff_model, dm)
    if diff_model.load_best_checkpoint():
        trainer.test(diff_model, dm)
    else:
        print("Best checkpoint was not found. Skipping test run.")


if __name__ == "__main__":
    main()
