import pathlib
import tempfile
from functools import partial

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
from src.data_module import data_process_scenario
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
        history,
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
        self.history = history
        self.samples = 10
        self.global_step_ = 0
        self.metrics_train = MetricCollection(
            [instantiate(m) for m in self.metrics.train]
        )
        self.metrics_val = MetricCollection([instantiate(m) for m in self.metrics.val])
        self.val_metric_batches = 3
        self._val_batches_for_metrics = []
        self.metrics_prefix = "Val"
        self.model = self.get_model(key_model)
        self.configure_ddpm_scheduler()
        self.configure_optimizers()

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
        traj = batch["traj"]
        context = batch["context"]
        mask = batch.get("mask", None)

        N = traj.shape[0]

        # flatten per agent
        traj = traj.reshape(N, -1)  # [N, H*2]
        context = context.reshape(N, -1)  # [N, T*2]

        mean = traj * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)

        key_agents = jr.split(key, N)
        noise = jax.vmap(lambda k: jr.normal(k, (traj.shape[1],)))(
            key_agents
        )  # [N, H*2]
        y = mean + std * noise  # [N, H*2]

        # shared model applied per agent
        pred = jax.vmap(lambda y_i, c_i: model(t, y_i, c_i))(y, context)  # [N, H*2]

        err = (pred + noise / std) ** 2

        if mask is not None:
            mask = jnp.repeat(mask[..., None], 2, axis=-1)  # [N, H, 2]
            mask = mask.reshape(N, -1).astype(err.dtype)  # [N, H*2]
            loss = (err * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        else:
            loss = jnp.mean(err)

        return weight(t) * loss

    @staticmethod
    def batch_loss_fn(model, weight, int_beta, batch, t1, key):

        batch_size = batch["traj"].shape[0]
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

    def on_fit_start(self) -> None:
        pathlib.Path("checkpoints").mkdir(exist_ok=True)
        print(self.hparams)
        if self.load_last_checkpoint:
            try:
                self.model = eqx.tree_deserialise_leaves(
                    f"checkpoints/ScoreBased/last.eqx", self.model
                )
                print("Loaded weights")
            except:
                print("Didnt load weights")

    # def on_fit_end(self):
    # pathlib.Path.mkdir(f"checkpoints/ScoreBased", parents=True, exist_ok=True)
    # eqx.tree_serialise_leaves(f"checkpoints/ScoreBased/last.eqx", self.model)

    def training_step(self, batch):
        batch.update(BaseDiffusionModel.get_data(batch, self.history))
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
        return dict_

    def sample(self, context, horizon, key=None):
        """
        Sample future trajectories for all agents.
        :param context: [N, T_hist, 2]
        :param horizon: int H
        :return: [N, H, 2]
        """
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        N = context.shape[0]
        context_flat = context.reshape(N, -1)  # [N, T_hist*2]
        sample_keys = jr.split(key, N)

        sample_one = lambda kk, cc: self.single_sample_fn(
            self.model, self.int_beta, (horizon * 2,), self.dt0, self.t1, kk, cc
        )
        pred_flat = jax.vmap(sample_one)(sample_keys, context_flat)  # [N, H*2]
        return pred_flat.reshape(N, horizon, 2)

    def sample_all_solutions(self, context, horizon, num_solutions=20, key=None):
        """
        Sample full reverse trajectory states for all agents.
        Returns:
            final_pred: [N, H, 2]
            all_preds: [S, N, H, 2] where S=num_solutions
        """
        if key is None:
            self.sample_key, key = jr.split(self.sample_key)

        N = context.shape[0]
        context_flat = context.reshape(N, -1)  # [N, T_hist*2]
        sample_keys = jr.split(key, N)

        sample_one = lambda kk, cc: self.single_sample_path_fn(
            self.model,
            self.int_beta,
            (horizon * 2,),
            self.dt0,
            self.t1,
            kk,
            cc,
            num_solutions,
        )
        # [N, S, H*2]
        pred_path_flat = jax.vmap(sample_one)(sample_keys, context_flat)
        pred_path = pred_path_flat.reshape(N, num_solutions, horizon, 2)  # [N, S, H, 2]
        pred_path = jnp.transpose(pred_path, (1, 0, 2, 3))  # [S, N, H, 2]
        final_pred = pred_path[-1]
        return final_pred, pred_path

    def validation_step(self, batch):
        batch.update(BaseDiffusionModel.get_data(batch, self.history))
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

    def on_validation_epoch_end(self) -> None:
        # pathlib.Path.mkdir(f"checkpoints/ScoreBased", parents=True, exist_ok=True)
        # eqx.tree_serialise_leaves(f"checkpoints/ScoreBased/last.eqx", self.model)
        if self.metrics_val is None:
            return

        images = []
        diffusion_video_frames = []
        self.metrics_val.reset()
        for batch in self._val_batches_for_metrics:
            gt_xy = batch["traj"]
            valid = batch["mask"]
            context = batch["context"]

            # Visualize/evaluate first batch element only (consistent with batch_idx=0 below)
            gt_xy_0 = gt_xy[0]  # [N, H, 2]
            valid_0 = valid[0]  # [N, H]
            context_0 = context[0]  # [N, T_hist, 2]
            _, H, _ = gt_xy_0.shape
            sample_steps = int(self.vis.get("sample_steps", 20))
            pred_xy, pred_path = self.sample_all_solutions(
                context_0, H, num_solutions=sample_steps
            )

            self.metrics_val.update(pred_xy, gt_xy_0, valid_0)

            viz_state = (
                batch["scenario"]
                if isinstance(batch, dict) and "scenario" in batch
                else batch
            )
            img = plot_simulator_state(
                viz_state, batch_idx=0, pred_xy=pred_xy, **self.vis
            )
            images.append(wandb.Image(img))

            # Build diffusion video on the first validation batch only.
            # if len(diffusion_video_frames) == 0 and viz_state is not None:
            #     pred_path_np = np.asarray(pred_path)
            #     for s in range(pred_path_np.shape[0]):
            #         frame = plot_simulator_state(
            #             viz_state,
            #             batch_idx=0,
            #             pred_xy=pred_path_np[s],
            #             **self.vis
            #         )
            #         diffusion_video_frames.append(frame)

        self.logger.log_image(
            key="Scenarios and predictions",
            images=images,
        )
        if len(diffusion_video_frames) > 0:
            video_np = np.stack(diffusion_video_frames, axis=0).astype(
                np.uint8
            )  # [T, H, W, C]
            gif_path = (
                pathlib.Path(tempfile.gettempdir())
                / f"diffusion_path_{self.current_epoch}.gif"
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

    @staticmethod
    @eqx.filter_jit
    def single_sample_fn(model, int_beta, data_shape, dt0, t1, key, context):
        """
        Sampling a single trajectory starting from normal noise at t1 and recovering data distribution at t0
        :param model:
        :param int_beta:
        :param data_shape:
        :param dt0:
        :param t1:
        :param key:
        :return:
        """

        def drift(t, y, args):
            """
            compute time derivative of function dß(t)/dt
            Noising SDE: dx(t) = -1/2 ß(t) x(t) dt + ß(t)^1/2 dW_t -> μ(x(t)) = - 1/2 ß(t) x(t) dt and σ^2 = ß(t)
            Reverse SDE: μ(x(tau)) = 1/2 ß(t) x(t) + ß(t) ∇ log p
            :param t:
            :param y:
            :param args:
            :return:
            """
            t = jnp.array(t)
            _, beta = jax.jvp(fun=int_beta, primals=(t,), tangents=(jnp.ones_like(t),))
            return (
                -0.5 * beta * (y + model(t, y, context))
            )  # negative because we use -dt0 when solving

        term = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        t0 = 0
        y1 = jr.normal(
            key, data_shape
        )  # noise at t1, from which integrate backwards to data distribution
        # reverse time, solve from t1 to t0
        sol = dfx.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t1,
            t1=t0,
            dt0=-dt0,
            y0=y1,
            # adjoint=dfx.NoAdjoint(),
        )
        return sol.ys[0]

    @staticmethod
    @eqx.filter_jit
    def single_sample_path_fn(
        model, int_beta, data_shape, dt0, t1, key, context, num_solutions
    ):
        """
        Return full reverse trajectory states sampled at `num_solutions` times.
        Output shape: [num_solutions, data_shape]
        """

        def drift(t, y, args):
            t = jnp.array(t)
            _, beta = jax.jvp(fun=int_beta, primals=(t,), tangents=(jnp.ones_like(t),))
            return -0.5 * beta * (y + model(t, y, context))

        term = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        t0 = 0.0
        y1 = jr.normal(key, data_shape)
        ts = jnp.linspace(t1, t0, int(num_solutions))
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
    def get_data(batch, history):
        # past_xy = batch["agents_history"][..., :history, :2]
        # future_xy = batch["agents_future"][..., :, :2]
        # valid = jnp.any(batch["agents_future"][..., :, :2] != 0, axis=-1)
        # return {
        #     "traj": future_xy,
        #     "context": past_xy,
        #     "mask": valid,
        # }
        traj = batch["scenario"].log_trajectory
        # past trajectory (context)
        past_xy = jnp.stack(
            [traj.x[..., :history], traj.y[..., :history]],
            axis=-1,
        )  # [B, N, T_hist, 2]

        # future trajectory (target for diffusion)
        future_xy = jnp.stack(
            [traj.x[..., history:], traj.y[..., history:]],
            axis=-1,
        )  # [B, N, H, 2]

        # mask for valid objects
        valid = traj.valid[..., history:]  # [B, N, H]

        return {
            "traj": future_xy,
            "context": past_xy,
            "mask": valid,
        }
