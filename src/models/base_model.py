import pathlib
from functools import partial

import diffrax as dfx
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch
import wandb
from hydra.utils import instantiate
from src.visualization.viz import plot_simulator_state


class BaseDiffusionModel(L.LightningModule):
    def __init__(
        self,
        seed,
        load_last_checkpoint,
        cfg_metrics,
        grad_clip,
        vis_cfg,
        cfg_model,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.key = jax.random.PRNGKey(seed)
        self.key, self.model_key, self.train_key, self.loader_key, self.sample_key = (
            jax.random.split(self.key, 5)
        )
        self.key, subkey = jax.random.split(self.key)
        self.cfg_model = cfg_model
        self.metrics = cfg_metrics
        self.vis = vis_cfg
        self.grad_clip = grad_clip
        self.load_last_checkpoint = load_last_checkpoint
        self.history = cfg_model.args.history
        self.t1 = 10.0
        self.int_beta = lambda t: t
        self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
        self.dt0 = 0.1
        self.samples = 10
        self.global_step_ = 0
        self.metrics = cfg_metrics
        self.model = self.get_model()
        self.val_metric_batches = 3
        self._val_batches_for_metrics = []
        self.configure_optimizers()

    def get_model(self):
        raise NotImplementedError(
            "Should not use base class. Should implement get_model for child class"
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "Should not use base class. Should implement configure_optimizers for child class"
        )

    def single_loss_fn(self):
        raise NotImplementedError(
            "Should not use base class. Should implement single_loss_fn for child class"
        )

    def batch_loss_fn(self):
        raise NotImplementedError(
            "Should not use base class. Should implement batch_loss_fn for child class"
        )

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
        """"TODO: implement sampling
        self.logger.log_image(
            key="Samples P1",
            images=[wandb.Image(self.sample(), caption="Samples P1")],
        )
        """

    # def on_fit_end(self):
    # pathlib.Path.mkdir(f"checkpoints/ScoreBased", parents=True, exist_ok=True)
    # eqx.tree_serialise_leaves(f"checkpoints/ScoreBased/last.eqx", self.model)

    def training_step(self, batch):
        data = self.get_data(batch, t=self.history)
        value, self.model, self.train_key, self.opt_state = (
            BaseDiffusionModel.make_step(
                self.model,
                self.batch_loss_fn,
                self.weight,
                self.int_beta,
                data,
                self.t1,
                self.train_key,
                self.opt_state,
                self.optim.update,
            )
        )
        dict_ = {"loss": torch.scalar_tensor(value.item())}
        self.log_dict(dict_, prog_bar=True)
        self.global_step_ += 1
        return dict_

    def sample(self):
        self.sample_key, *sample_key = jr.split(self.sample_key, self.samples**2 + 1)
        sample_key = jnp.stack(sample_key)
        sample_fn = partial(
            BaseDiffusionModel.single_sample_fn,
            self.model,
            self.int_beta,
            (1, 28, 28),
            self.dt0,
            self.t1,
        )
        sample = jax.vmap(sample_fn)(sample_key)
        # sample = data_mean + data_std * sample
        # sample = jnp.clip(sample, data_min, data_max)
        sample = einops.rearrange(
            sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=self.samples, n2=self.samples
        )
        fig = plt.figure()
        plt.imshow(sample, cmap="Greys")
        plt.axis("off")
        plt.title(f"{self.global_step_}")
        plt.tight_layout()
        if self.hparams.get("show", False):
            plt.show()
        return fig

    def validation_step(self, batch):
        data = self.get_data(batch, t=self.history)
        value = self.batch_loss_fn(self.model, self.weight, self.int_beta, data, self.t1, self.loader_key)
        # dict_ = {"loss": torch.scalar_tensor(value.item())}
        self.log("Val_Loss", jnp.asarray(value).item(), prog_bar=True, batch_size=1)
        # collect some first batches to compute metrics on
        if (
            self.metrics is not None
            and len(self._val_batches_for_metrics) < self.val_metric_batches
        ):
            self._val_batches_for_metrics.append(batch)

    def on_validation_epoch_end(self) -> None:
        # pathlib.Path.mkdir(f"checkpoints/ScoreBased", parents=True, exist_ok=True)
        # eqx.tree_serialise_leaves(f"checkpoints/ScoreBased/last.eqx", self.model)
        if self.metrics is None:
            return
        #TODO: traj sampling
        """
        images = []
        self.metrics.reset()
        for batch in self._val_batches_for_metrics:
            # Assuming batch = {"gt_xy": gt_xy, "gt_valid": gt_valid, ...}
            data = self.get_data(batch, t=0)
            gt_xy = data["traj"]
            valid = data["mask"]
            
            self.sample_key, k = jr.split(self.sample_key)
            pred_xy = self.sample_trajectory(batch, k)  # must match gt shape

            self.metrics.update(pred_xy, gt_xy, valid)

            img = plot_simulator_state(
                batch,
                batch_idx=0,
                use_log_traj=self.cfg_vis.get("use_log_traj", True),
                viz_config=self.cfg_vis.get("viz_config"),
                plot_all_trajectories=self.cfg_vis.get("plot_all_trajectories", True),
                pred_xy=pred_xy,
                pred_alpha=self.cfg_vis.get("pred_alpha", 0.8),
                pred_linewidth=self.cfg_vis.get("pred_linewidth", 2.0),
                pred_linestyle=self.cfg_vis.get("pred_linestyle", "--"),
                pred_clip_to_view=self.cfg_vis.get("pred_clip_to_view", True),
                gt_linewidth=self.cfg_vis.get("gt_linewidth", 2.0),
            )
            images.append(wandb.Image(img))

        self.logger.log_image(
            key="Scenarios and predictions",
            images=images,
        )
        
        vals = self.metrics.compute()
        log_dict = {
            f"{self.metrics_prefix}/{k}": float(jnp.asarray(v)) for k, v in vals.items()
        }
        self.log_dict(log_dict, prog_bar=True)
        """
        self._val_batches_for_metrics.clear()
        

    @staticmethod
    @eqx.filter_jit
    def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
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
                -0.5 * beta * (y + model(t, y))
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
    def make_step(model, batch_loss_fn, weight, int_beta, data, t1, key, opt_state, opt_update):
        loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
        loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
        updates, opt_state = opt_update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        key = jr.split(key, 1)[0]
        return loss, model, key, opt_state
