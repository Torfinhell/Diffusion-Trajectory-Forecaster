from pathlib import Path

import jax

from src.trainers.base_trainer import BaseTrainer


class BaseProfilerDebug(BaseTrainer):
    def __init__(
        self,
        model,
        loss_fn,
        optim,
        opt_state,
        diffusion_sampler,
        cfg_metrics,
        vis_cfg,
        key,
        train_key,
        loader_key,
        sample_key,
        log_dir,
        start_step,
        num_steps,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optim=optim,
            opt_state=opt_state,
            diffusion_sampler=diffusion_sampler,
            cfg_metrics=cfg_metrics,
            vis_cfg=vis_cfg,
            key=key,
            train_key=train_key,
            loader_key=loader_key,
            sample_key=sample_key,
            **kwargs,
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_server(9999)
        self.start_step = int(start_step)
        self.stop_step = self.start_step + int(num_steps)

    def training_step(self, batch, batch_idx):
        if self.start_step <= self.global_step_ <= self.stop_step:
            with jax.profiler.StepTraceAnnotation("train", step_num=self.global_step_):
                self._step(batch, "train")
        else:
            self._step(batch, "train")
        return None

    def validation_step(self, batch, batch_idx):
        with jax.profiler.StepTraceAnnotation("val", step_num=self.global_step_):
            return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        with jax.profiler.StepTraceAnnotation("test", step_num=self.global_step_):
            return self._step(batch, "test")
