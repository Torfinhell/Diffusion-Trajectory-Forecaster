import numpy as np
from pytorch_lightning.loggers.logger import Logger

from clearml import Task


class ClearMLLogger(Logger):
    def __init__(self, mode="online", **kwargs):
        super().__init__()
        Task.set_offline(mode == "offline")
        self._task = Task.init(**kwargs)
        self._clearml_logger = self._task.get_logger()
        self._global_step = 0

    @property
    def experiment(self) -> Task:
        return self._task

    @property
    def name(self) -> str:
        return self._task.project

    @property
    def version(self) -> str:
        return self._task.id

    # NOTE: no @rank_zero_only -- training is single-GPU here, and the decorator
    # silently no-ops ALL logging if a stray RANK/LOCAL_RANK/SLURM_PROCID env var
    # is nonzero (common on cloud/k8s/SLURM hosts), which produced "task created
    # but no scalars/plots" on remote machines.
    def log_metrics(self, metrics, step=None):
        if step is not None and step > self._global_step:
            self._global_step = step
        else:
            self._global_step += 1
        for k, v in metrics.items():
            self._clearml_logger.report_scalar("metrics", k, v, self._global_step)

    def log_hyperparams(self, params, *args, **kwargs):
        self._task.connect(params)

    def log_image(self, key, images, step=None):
        for i, img in enumerate(images):
            self._clearml_logger.report_image(
                title=key,
                series=str(i),
                iteration=step or 0,
                image=np.asarray(img),
            )

    def upload_artifact(self, name, path, metadata=None):
        self._task.upload_artifact(name, str(path), metadata=metadata)

    def save(self) -> None:
        # Force-flush buffered scalars. ClearML batches scalar reports and sends
        # them on a background timer; artifacts upload eagerly. On an ephemeral /
        # killed remote run the process can exit before the timer's final flush,
        # so checkpoints (artifacts) survive but scalars are silently lost. Lightning
        # calls save() at checkpoint/epoch boundaries -- flush here so scalars land.
        if self._clearml_logger is not None:
            self._clearml_logger.flush()

    def finalize(self, status: str) -> None:
        if self._clearml_logger is not None:
            self._clearml_logger.flush()
        if self._task is not None:
            self._task.flush(wait_for_uploads=True)
