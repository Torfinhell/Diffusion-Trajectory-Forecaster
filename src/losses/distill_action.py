from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.data_module.agent_path import AgentPath


def model_feature_dims(model_cfg: Any) -> dict[str, int | list[int]]:
    return {
        "kv_cond": int(model_cfg.se_args.out_dim),
        "sa": [int(model_cfg.samlp_args.out_dim)] * int(model_cfg.num_sa_mlp),
        "ca": [int(model_cfg.camlp_args.out_dim)] * int(model_cfg.num_camlp),
    }


def _build_projectors(
    lambdas: dict,
    student_dims: dict[str, int | list[int]],
    teacher_dims: dict[str, int | list[int]],
    key,
) -> dict[str, list[eqx.nn.Linear | None]]:
    active = [k for k, lam in lambdas.items() if k not in ("gt", "out") and lam != 0.0]
    if not active:
        return {}
    feat_keys = jr.split(key, len(active))
    projectors = {}
    for feat_key, feat_key_rng in zip(active, feat_keys):
        s_dims = student_dims[feat_key]
        t_dims = teacher_dims[feat_key]
        if isinstance(s_dims, int):
            s_dims, t_dims = [s_dims], [t_dims]
        layer_keys = jr.split(feat_key_rng, len(s_dims))
        projectors[feat_key] = [
            None if s == t else eqx.nn.Linear(s, t, use_bias=False, key=k)
            for s, t, k in zip(s_dims, t_dims, layer_keys)
        ]
    return projectors


def _project(proj: eqx.nn.Linear | None, feat: jnp.ndarray) -> jnp.ndarray:
    if proj is None:
        return feat
    return jax.vmap(proj)(feat)


class KDLoss(eqx.Module):
    teacher: Optional[eqx.Module]
    projectors: dict
    lambdas: dict
    accel_scale: float
    yaw_rate_scale: float
    perturbation_per_sample: int
    perturbation_std: float

    def __init__(
        self,
        lambdas: dict,
        teacher: eqx.Module | None = None,
        student_dims: dict[str, int | list[int]] | None = None,
        teacher_dims: dict[str, int | list[int]] | None = None,
        key=None,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
        perturbation_per_sample: int = 1,
        perturbation_std: float = 0.0,
    ):
        self.lambdas = lambdas
        self.teacher = teacher
        self.accel_scale = accel_scale
        self.yaw_rate_scale = yaw_rate_scale
        self.perturbation_per_sample = max(1, int(perturbation_per_sample))
        self.perturbation_std = float(perturbation_std)
        if teacher is None:
            self.projectors = {}
        else:
            if student_dims is None or teacher_dims is None or key is None:
                raise ValueError(
                    "student_dims, teacher_dims, and key are required when teacher is set"
                )
            self.projectors = _build_projectors(
                lambdas, student_dims, teacher_dims, key
            )

    def _loss_one_draw(
        self,
        model,
        diffusion_sampler,
        gt_actions,
        gt_actions_norm,
        past_path: AgentPath,
        future_path: AgentPath,
        key,
        **kwargs,
    ):
        if self.perturbation_std > 0.0:
            perturb_key, timestep_key, noise_key = jr.split(key, 3)
            x0 = gt_actions_norm + (
                jr.normal(perturb_key, gt_actions_norm.shape) * self.perturbation_std
            )
        else:
            timestep_key, noise_key = jr.split(key)
            x0 = gt_actions_norm

        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_actions.shape)
        noisy_actions = diffusion_sampler.add_noise(x0, noise, timestep)

        student_out, student_feats = model.__call_with_features__(
            timestep, noisy_actions, **kwargs
        )

        teacher_out = None
        teacher_feats = None
        if self.teacher is not None:
            teacher_out, teacher_feats = self.teacher.__call_with_features__(
                timestep, noisy_actions, **kwargs
            )
            teacher_out = jax.lax.stop_gradient(teacher_out)
            teacher_feats = jax.lax.stop_gradient(teacher_feats)

        pred_xy = past_path.decode_action_sample(
            student_out,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        gt_xy = past_path.trajectory_from_anchor(future_path)
        err = (pred_xy - gt_xy) ** 2
        valid_target = future_path.valid_mask
        if valid_target is None:
            valid_target = jnp.ones((future_path.path.shape[0],), dtype=bool)
        if valid_target.ndim == err.ndim - 1:
            valid_target = valid_target[..., None]
        weights = jnp.asarray(valid_target, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        l_gt = (err * weights).sum() / jnp.maximum(weights.sum(), 1.0)

        stats = {"L_gt": l_gt}
        total = self.lambdas["gt"] * l_gt

        if self.teacher is None:
            return total, stats

        projectors = self.projectors
        for feat_key, lam in self.lambdas.items():
            if feat_key == "gt" or lam == 0.0:
                continue
            if feat_key == "out":
                loss_term = jnp.mean((student_out - teacher_out) ** 2)
            elif feat_key == "kv_cond":
                projs = projectors[feat_key]
                loss_term = jnp.mean(
                    (
                        _project(projs[0], student_feats[feat_key])
                        - teacher_feats[feat_key]
                    )
                    ** 2
                )
            else:
                projs = projectors[feat_key]
                t_list = teacher_feats[feat_key]
                s_list = student_feats[feat_key]
                if len(t_list) == 0:
                    loss_term = jnp.zeros(())
                else:
                    layer_losses = [
                        jnp.mean((_project(projs[i], s_list[i]) - t_list[i]) ** 2)
                        for i in range(len(t_list))
                    ]
                    loss_term = sum(layer_losses) / len(layer_losses)
            stats[f"L_{feat_key}"] = loss_term
            total = total + lam * loss_term
        return total, stats

    def __call__(
        self,
        model,
        diffusion_sampler,
        past_path: AgentPath,
        future_path: AgentPath,
        key,
        debug=False,
        agent_coeff=None,
        **kwargs,
    ):
        valid = future_path.valid_mask
        if valid is None:
            valid = jnp.ones((future_path.path.shape[0],), dtype=bool)
        gt_actions, _ = future_path.actions(valid)
        action_scale = jnp.asarray(
            [self.accel_scale, self.yaw_rate_scale], dtype=gt_actions.dtype
        )
        gt_actions_norm = gt_actions / action_scale

        n = self.perturbation_per_sample
        if n == 1:
            total, stats = self._loss_one_draw(
                model,
                diffusion_sampler,
                gt_actions,
                gt_actions_norm,
                past_path,
                future_path,
                key,
                **kwargs,
            )
        else:
            keys = jr.split(key, n)

            def one_draw(draw_key):
                return self._loss_one_draw(
                    model,
                    diffusion_sampler,
                    gt_actions,
                    gt_actions_norm,
                    past_path,
                    future_path,
                    draw_key,
                    **kwargs,
                )

            totals, stats = jax.vmap(one_draw)(keys)
            total = jnp.mean(totals)
            stats = jax.tree.map(lambda x: jnp.mean(x, axis=0), stats)

        loss_dict = {"loss": total}
        if debug:
            loss_dict.update(stats)
        # scale by agent coefficient if provided
        agent_coeff = (
            kwargs.pop("agent_coeff", agent_coeff)
            if "agent_coeff" in kwargs
            else agent_coeff
        )
        if agent_coeff is not None:
            loss_dict = jax.tree_map(
                lambda v: v * jnp.asarray(agent_coeff, dtype=v.dtype), loss_dict
            )
        return loss_dict
