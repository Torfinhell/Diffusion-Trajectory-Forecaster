from pathlib import Path

import jax
from pytorch_lightning.callbacks import Callback


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

    def on_test_end(self, trainer, pl_module):
        del pl_module
        self._maybe_flush(trainer, force=True)


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
