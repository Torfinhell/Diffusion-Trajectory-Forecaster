from pathlib import Path

import jax
from pytorch_lightning.callbacks import Callback


class JaxProfilerCallback(Callback):
    """Profile the first `limit_profile_batches` train/val batches once, then upload traces."""

    def __init__(self, log_dir: str, limit_profile_batches: int = 3):
        self.log_dir = Path(log_dir)
        self.limit_profile_batches = int(limit_profile_batches)
        self._train_done = False
        self._val_done = False
        self._active_stage = None

    def _trace_dir(self, stage: str) -> Path:
        return self.log_dir / stage

    def _start(self, stage: str):
        trace_dir = self._trace_dir(stage)
        trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(trace_dir))
        self._active_stage = stage

    def _stop_and_upload(self, trainer, stage: str):
        jax.profiler.stop_trace()
        self._active_stage = None
        logger = trainer.logger
        trace_dir = self._trace_dir(stage)
        if logger is not None and hasattr(logger, "upload_artifact"):
            logger.upload_artifact(
                name=f"jax_profiler_{stage}",
                path=trace_dir,
                metadata={
                    "stage": stage,
                    "limit_profile_batches": self.limit_profile_batches,
                },
            )

    def on_train_epoch_start(self, trainer, pl_module):
        if self._train_done:
            return
        self._start("train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._train_done or self._active_stage != "train":
            return
        if batch_idx + 1 >= self.limit_profile_batches:
            self._stop_and_upload(trainer, "train")
            self._train_done = True

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._val_done:
            return
        self._start("val")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self._val_done or self._active_stage != "val":
            return
        if batch_idx + 1 >= self.limit_profile_batches:
            self._stop_and_upload(trainer, "val")
            self._val_done = True
