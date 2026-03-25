import tempfile
from functools import partial
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from PIL import Image

import wandb
from src.data_module import data_process_scenarios
from src.metrics import MetricCollection
from src.visualization.viz import plot_simulator_state


class BaseDiffusionModel(L.LightningModule):
    def __init__(
        self,
        seed,
        load_last_checkpoint,
        cfg_metrics,
        grad_clip,
        vis_cfg,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.key = jax.random.PRNGKey(seed)
        self.key, key_model, self.train_key, self.loader_key, self.sample_key = (
            jax.random.split(self.key, 5)
        )
        self.metrics = cfg_metrics
        self.vis = vis_cfg
        self.grad_clip = grad_clip
        self.load_last_checkpoint = load_last_checkpoint
        self.samples = 10
        self.global_step_ = 0  # TODO
        self.metrics_train = MetricCollection(
            [instantiate(m) for m in self.metrics.train]
        )
        self.metrics_val = MetricCollection([instantiate(m) for m in self.metrics.val])
        self.val_metric_batches = 3
        self._val_batches_for_metrics = []
        self.train_metric_batches = 3
        self._train_batches_for_metrics = []
        self.metrics_prefix = "Val"
        self.model = self.get_model(key_model)
        self.configure_ddpm_scheduler()
        self.configure_optimizers()
        self.data_shape = None

    def get_model(self, cfg_model, key_model):
        raise NotImplementedError(
            "Should not use base class. Should implement get_model for child class"
        )

    def configure_ddpm_scheduler(self):
        raise NotImplementedError(
            "Should not use base class. Should implement configure_ddpm_scheduler for child class"
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "Should not use base class. Should implement configure_optimizers for child class"
        )

    def training_step(self, batch):
        value, self.model, self.train_key, self.opt_state = (
            BaseDiffusionModel.make_step(
                self.model,
                self.batch_loss_fn,
                self.weight,
                self.int_beta,
                batch,
                self.t1,
                self.train_key,
                self.opt_state,
                self.optim.update,
            )
        )
        loss_value = jnp.asarray(value).item()
        self.log(
            "Train_Loss",
            loss_value,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        dict_ = {"loss": torch.scalar_tensor(loss_value)}
        self.global_step_ += 1
        if len(self._train_batches_for_metrics) < self.train_metric_batches:
            self._train_batches_for_metrics.append(batch)
        return dict_

    def validation_step(self, batch):
        self.loader_key, val_key = jr.split(self.loader_key)
        value = self.batch_loss_fn(
            self.model, self.weight, self.int_beta, batch, self.t1, val_key
        )
        self.log(
            "Val_Loss",
            jnp.asarray(value).item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        # collect some first batches to compute metrics on
        if (
            self.metrics is not None
            and len(self._val_batches_for_metrics) < self.val_metric_batches
        ):
            self._val_batches_for_metrics.append(batch)

    @staticmethod
    @eqx.filter_jit
    def make_step(
        model, batch_loss_fn, weight, int_beta, batch, t1, key, opt_state, opt_update
    ):
        loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
        loss, grads = loss_fn(model, weight, int_beta, batch, t1, key)
        updates, opt_state = opt_update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        key = jr.split(key, 1)[0]
        return loss, model, key, opt_state

    @staticmethod
    def batch_loss_fn(model, weight, int_beta, batch, t1, key):

        batch_size = batch["gt_xy"].shape[0]
        tkey, losskey = jr.split(key)
        losskey = jr.split(losskey, batch_size)
        """
		Low-discrepancy sampling over t to reduce variance
		by sampling very evenly by sampling uniformly and independently from (t1-t0)/batch_size bins
		t = [U(0,1), U(1,2), U(2,3), ...]
		"""
        t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
        t = t + (t1 / batch_size) * jnp.arange(batch_size)
        """ Fixing the first three arguments of single_loss_fn, leaving batch, t and key as input """
        loss_fn = partial(BaseDiffusionModel.single_loss_fn, model, weight, int_beta)
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(batch, t, losskey))

    @staticmethod
    def single_loss_fn(model, weight, int_beta, batch, t, key):
        """
        OU process provides analytical mean and variance
        int_beta(t) = ß = θ
        E[X_t] = μ + exp[-θ t] ( X_0 - μ) w/ μ=0 gives =X_0 * exp[ - θ t ]
        V[X_t] = σ^2/(2θ) ( 1 - exp(-2 θ t) ) w/ σ^2=ß=θ gives = 1 - exp(-2 ß t)
        :param model:
        :param weight:
        :param int_beta:
        :param batch:
        :param t:
        :param key:
        :return:
        """
        INPUT_DIM = batch["gt_xy"].shape
        mean = batch["gt_xy"] * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, INPUT_DIM)
        y = mean + std * noise
        # shared model applied per agent
        pred = model(t, y, batch["context"])
        err = (pred + noise / std) ** 2
        loss = (err * batch["gt_xy_mask"]).sum() / jnp.maximum(
            batch["gt_xy_mask"].sum(), 1.0
        )
        return weight(t) * loss

    def on_fit_start(self) -> None:
        Path("checkpoints").mkdir(exist_ok=True)
        print(self.hparams)
        if self.load_last_checkpoint:
            try:
                self.model = eqx.tree_deserialise_leaves(
                    f"checkpoints/ScoreBased/last.eqx", self.model
                )
                print("Loaded weights")
            except:
                print("Didnt load weights")

    def on_fit_end(self):
        if not Path(f"checkpoints/ScoreBased").exists():
            Path.mkdir(f"checkpoints/ScoreBased", parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(f"checkpoints/ScoreBased/last.eqx", self.model)

    def on_train_epoch_end(self) -> None:
        if len(self.metrics_train) == 0 or len(self._train_batches_for_metrics) == 0:
            return
        self.metrics_train.reset()
        for batch in self._train_batches_for_metrics:
            self._update_metrics_for_batch(self.metrics_train, batch)
        vals = self.metrics_train.compute()
        log_dict = {f"Train/{k}": float(jnp.asarray(v)) for k, v in vals.items()}
        self.log_dict(log_dict, prog_bar=True)
        print(f"[Train metrics] epoch={self.current_epoch}  " + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
        self._train_batches_for_metrics.clear()

    def on_validation_epoch_end(self) -> None:
        if not Path(f"checkpoints/ScoreBased").exists():
            Path.mkdir(f"checkpoints/ScoreBased", parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(f"checkpoints/ScoreBased/last.eqx", self.model)
        if len(self.metrics_val) == 0 or len(self._val_batches_for_metrics) == 0:
            return

        images = []
        diffusion_video_frames = []
        self.metrics_val.reset()
        for batch_idx, batch in enumerate(self._val_batches_for_metrics):
            first_pred_xy, first_pred_path = self._update_metrics_for_batch(
                self.metrics_val,
                batch,
                return_first_prediction=batch_idx == 0,
            )

            if batch_idx == 0 and "scenario" in batch and first_pred_xy is not None:
                viz_state = batch["scenario"]
                if viz_state is not None:
                    img = plot_simulator_state(
                        viz_state, batch_idx=0, pred_xy=first_pred_xy, **self.vis
                    )
                    images.append(wandb.Image(img))
                    if len(diffusion_video_frames) == 0 and first_pred_path is not None:
                        pred_path_np = np.asarray(first_pred_path)
                        for s in range(pred_path_np.shape[0]):
                            frame = plot_simulator_state(
                                viz_state,
                                batch_idx=0,
                                pred_xy=pred_path_np[s],
                                **self.vis,
                            )
                            diffusion_video_frames.append(frame)
        if "scenario" in batch:
            self.logger.log_image(
                key="Scenarios and predictions",
                images=images,
            )
        if len(diffusion_video_frames) > 0:
            video_np = np.stack(diffusion_video_frames, axis=0).astype(
                np.uint8
            )  # [T, H, W, C]
            gif_path = (
                Path(tempfile.gettempdir()) / f"diffusion_path_{self.current_epoch}.gif"
            )
            pil_frames = [Image.fromarray(frame) for frame in video_np]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(1000 / int(self.vis.get("sample_video_fps", 6)))),
                loop=0,
            )
            self.logger.experiment.log(
                {
                    "Diffusion trajectory video": wandb.Video(
                        str(gif_path),
                        fps=int(self.vis.get("sample_video_fps", 6)),
                        format="gif",
                    )
                }
            )

        vals = self.metrics_val.compute()
        log_dict = {
            f"{self.metrics_prefix}/{k}": float(jnp.asarray(v)) for k, v in vals.items()
        }
        self.log_dict(log_dict, prog_bar=True)
        self._val_batches_for_metrics.clear()

    def _update_metrics_for_batch(
        self, metrics, batch, return_first_prediction=False
    ):
        first_pred_xy = None
        first_pred_path = None
        num_solutions = int(self.vis.get("sample_steps", 20))

        for sample_idx in range(batch["gt_xy"].shape[0]):
            pred_xy, pred_path = self.sample_multiple_sol(
                batch["context"][sample_idx],
                num_solutions=num_solutions,
                predict_shape=batch["gt_xy"][sample_idx].shape,
            )
            metrics.update(
                pred_xy, batch["gt_xy"][sample_idx], batch["gt_xy_mask"][sample_idx]
            )
            if return_first_prediction and sample_idx == 0:
                first_pred_xy = pred_xy
                first_pred_path = pred_path

        return first_pred_xy, first_pred_path

    def sample_multiple_sol(
        self, context, num_solutions=20, predict_shape=None, save_full=False
    ):
        """
        Sample full reverse trajectory states for all agents.
        Returns:
            final_pred: [N, H, 2]
            all_preds: [S, N, H, 2] where S=num_solutions
        """
        self.sample_key, key = jr.split(self.sample_key)
        sample_keys = jr.split(key, num_solutions)
        sample_one = lambda kk: self.sample_one_sol(
            self.model,
            self.int_beta,
            predict_shape,
            self.dt0,
            self.t1,
            context,
            save_full,
            kk,
        )[0]
        pred_paths = jax.vmap(sample_one)(sample_keys)  # [S, *data_shape]
        final_pred = jnp.mean(pred_paths, axis=0)       # mean over samples → [*data_shape]
        return final_pred, pred_paths

    @eqx.filter_jit
    def sample_one_sol(
        self, model, int_beta, data_shape, dt0, t1, context, save_full=False, key=None
    ):
        """
        Return full reverse trajectory states sampled at `num_solutions` times.
        Output shape: [num_solutions, data_shape]
        """
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        def drift(t, y, args):
            t = jnp.array(t)
            _, beta = jax.jvp(fun=int_beta, primals=(t,), tangents=(jnp.ones_like(t),))
            return -0.5 * beta * (y + model(t, y, context))

        term = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        t0 = 0.0
        y1 = jr.normal(key, data_shape)
        if save_full:
            ts = jnp.linspace(t1, t0)
        else:
            ts = jnp.array([t0])
        sol = dfx.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t1,
            t1=t0,
            dt0=-dt0,
            y0=y1,
            saveat=dfx.SaveAt(ts=ts),
        )
        return sol.ys
