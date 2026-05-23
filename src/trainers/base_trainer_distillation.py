import equinox as eqx
import optax

from src.trainers.base_trainer import BaseTrainer


class BaseTrainerDistillation(BaseTrainer):
    def _init_trainer_state(self, cfg_metrics, vis_cfg, trainer_cfg):
        super()._init_trainer_state(
            cfg_metrics,
            vis_cfg,
            {**trainer_cfg, "loss_returns_stats": True},
        )

    def _apply_step_updates(self, step_out):
        if "projectors" in step_out:
            self.loss_fn = eqx.tree_at(
                lambda loss: loss.projectors,
                self.loss_fn,
                step_out["projectors"],
            )

    @staticmethod
    def make_step(
        model,
        diffusion_sampler,
        loss_fn,
        batch,
        key,
        train,
        opt_state=None,
        opt_update=None,
        return_loss_stats=False,
    ):
        if train:

            def packed_loss_fn(params):
                model_, projectors_ = params
                loss_fn_ = eqx.tree_at(
                    lambda loss: loss.projectors, loss_fn, projectors_
                )
                return BaseTrainer.batch_loss_fn(
                    model_,
                    diffusion_sampler,
                    loss_fn_,
                    batch,
                    key,
                    return_loss_stats=True,
                )

            grad_fn = eqx.filter_value_and_grad(packed_loss_fn, has_aux=True)
            projectors = loss_fn.projectors
            (loss, stats), (model_grads, proj_grads) = grad_fn((model, projectors))
            packed_grads = (model_grads, proj_grads)
            grad_norm = optax.global_norm(packed_grads)
            packed_updates, opt_state = opt_update(packed_grads, opt_state)
            model_updates, proj_updates = packed_updates
            update_norm = optax.global_norm(packed_updates)
            model = eqx.apply_updates(model, model_updates)
            projectors = eqx.apply_updates(projectors, proj_updates)
            param_norm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
        else:
            loss, stats = BaseTrainer.batch_loss_fn(
                model,
                diffusion_sampler,
                loss_fn,
                batch,
                key,
                return_loss_stats=True,
            )
            grad_norm = None
            update_norm = None
            param_norm = None
            projectors = None

        step_out = {
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "loss": loss,
            **stats,
        }
        if train:
            step_out["model"] = model
            step_out["opt_state"] = opt_state
            step_out["projectors"] = projectors
        return step_out
