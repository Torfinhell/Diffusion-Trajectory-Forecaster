from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lightning_fabric.utilities.logger import (
    _add_prefix,
    _convert_json_serializable,
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
)
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@dataclass(slots=True)
class _ScalarEvent:
    title: str
    series: str
    value: float
    iteration: int


class ClearMLLogger(Logger):
    LOGGER_JOIN_CHAR = "/"

    def __init__(
        self,
        project_name: str,
        task_name: str,
        save_dir: str = "clearml",
        mode: str = "online",
        output_uri: Optional[str] = None,
        prefix: str = "",
        flush_metrics_every_n_epochs: int = 10,
        metric_buffer_max_size: int = 1000,
    ) -> None:
        super().__init__()
        self._project_name = project_name
        self._task_name = task_name
        self._save_dir = save_dir
        self._mode = str(mode).lower()
        self._output_uri = output_uri
        self._prefix = prefix
        self._flush_metrics_every_n_epochs = max(1, int(flush_metrics_every_n_epochs))
        self._metric_buffer_max_size = max(1, int(metric_buffer_max_size))
        self._task = None
        self._clearml_logger = None
        self._pending_scalars: list[_ScalarEvent] = []
        self._last_flushed_epoch = -1

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._mode == "disabled":
            return None
        if self._task is not None:
            return self._task

        try:
            from clearml import Task
        except ImportError as exc:
            raise ModuleNotFoundError(
                "ClearML is not installed. Add `clearml` to the environment."
            ) from exc

        if self._mode == "offline":
            Task.set_offline(offline_mode=True)

        self._task = Task.init(
            project_name=self._project_name,
            task_name=self._task_name,
            output_uri=self._output_uri,
            reuse_last_task_id=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            auto_connect_streams=False,
        )
        self._clearml_logger = self._task.get_logger()
        return self._task

    @property
    def name(self) -> str:
        return self._project_name

    @property
    def version(self) -> str:
        task = self.experiment
        if task is None:
            return "disabled"
        return str(task.id)

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @rank_zero_only
    def log_hyperparams(
        self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any
    ) -> None:
        task = self.experiment
        if task is None:
            return
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        params = _flatten_dict(params)
        params = _convert_json_serializable(params)
        task.connect(params, name="hyperparameters")

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if self._mode == "disabled":
            return

        prefixed_metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        iteration = 0 if step is None else int(step)
        for key, value in prefixed_metrics.items():
            title, series = self._split_key(key)
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            self._pending_scalars.append(
                _ScalarEvent(
                    title=title,
                    series=series,
                    value=scalar,
                    iteration=iteration,
                )
            )

        if len(self._pending_scalars) >= self._metric_buffer_max_size:
            self.flush_metrics()

    @rank_zero_only
    def maybe_flush_metrics(self, epoch: int, force: bool = False) -> None:
        if not self._pending_scalars:
            return
        if force:
            self.flush_metrics()
            self._last_flushed_epoch = int(epoch)
            return

        epoch = int(epoch)
        if epoch <= self._last_flushed_epoch:
            return
        if epoch % self._flush_metrics_every_n_epochs != 0:
            return
        self.flush_metrics()
        self._last_flushed_epoch = epoch

    @rank_zero_only
    def flush_metrics(self) -> None:
        if not self._pending_scalars:
            return

        task = self.experiment
        if task is None or self._clearml_logger is None:
            self._pending_scalars.clear()
            return

        for event in self._pending_scalars:
            self._clearml_logger.report_scalar(
                title=event.title,
                series=event.series,
                value=event.value,
                iteration=event.iteration,
            )
        self._pending_scalars.clear()

    @rank_zero_only
    def log_image(self, key: str, images: list[Any], step: Optional[int] = None) -> None:
        task = self.experiment
        if task is None or self._clearml_logger is None:
            return

        iteration = 0 if step is None else int(step)
        for idx, image in enumerate(images):
            self._clearml_logger.report_image(
                title=key,
                series=str(idx),
                iteration=iteration,
                image=self._normalize_image(image),
            )

    @rank_zero_only
    def log_video(self, key: str, path: str | Path, step: Optional[int] = None) -> None:
        task = self.experiment
        if task is None or self._clearml_logger is None:
            return
        self._clearml_logger.report_media(
            title=key,
            series="0",
            iteration=0 if step is None else int(step),
            local_path=str(path),
        )

    @rank_zero_only
    def upload_artifact(
        self,
        name: str,
        path: str | Path,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        task = self.experiment
        if task is None:
            return
        upload_kwargs: dict[str, Any] = {
            "name": name,
            "artifact_object": str(path),
        }
        if metadata is not None:
            upload_kwargs["metadata"] = dict(metadata)
        task.upload_artifact(**upload_kwargs)

    @rank_zero_only
    def log_run_metadata(self, metadata: Mapping[str, Any]) -> None:
        task = self.experiment
        if task is None:
            return
        task.connect(_convert_json_serializable(dict(metadata)), name="run_metadata")

    @rank_zero_only
    def finalize(self, status: str) -> None:
        task = self._task
        if task is None:
            return
        self.flush_metrics()
        # ClearML waits up to 300s on shutdown for async repository / requirements
        # detection. For our runs this metadata is not worth stalling process exit,
        # especially in offline mode, so kill the background detection thread first.
        wait_for_repo_detection = getattr(task, "_wait_for_repo_detection", None)
        if callable(wait_for_repo_detection):
            wait_for_repo_detection(timeout=-1)
        task.close()

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        if "/" not in key:
            return "metrics", key
        title, series = key.rsplit("/", 1)
        return title, series

    @staticmethod
    def _normalize_image(image: Any) -> Any:
        if isinstance(image, np.ndarray):
            return image
        if hasattr(image, "__array__"):
            return np.asarray(image)
        return image
