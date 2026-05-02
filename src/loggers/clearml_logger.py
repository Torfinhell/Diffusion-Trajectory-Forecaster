import numpy as np
from clearml import Task
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ClearMLLogger(Logger):
    def __init__(self, project, task, mode="online", **kwargs):
        super().__init__()
        Task.set_offline(mode == "offline")
        self._task = Task.init(project, task, **kwargs)
        self._clearml_logger = self._task.get_logger()

    @property
    def experiment(self) -> Task:
        return self._task

    @property
    def name(self) -> str:
        return self._task.project

    @property
    def version(self) -> str:
        return self._task.id

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            self._clearml_logger.report_scalar("metrics", k, v, step or 0)

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        self._task.connect(params)

    @rank_zero_only
    def log_image(self, key, images, step=None):
        for i, img in enumerate(images):
            self._clearml_logger.report_image(
                title=key,
                series=str(i),
                iteration=step or 0,
                image=np.asarray(img),
            )

    @rank_zero_only
    def upload_artifact(self, name, path, metadata=None):
        self._task.upload_artifact(name, str(path), metadata=metadata)
