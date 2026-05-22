from collections.abc import Iterable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.utils.data_utils import predictions_to_local_xy

# feat_key -> (tensor name in feature dict, projector method, lambda key)
DISTILL_FEATURE_SPECS = {
    "kv_cond": ("kv_cond", "project_kv_cond", "scene"),
    "sa": ("sa_features", "project_sa", "sa"),
    "ca": ("ca_features", "project_ca", "ca"),
    "out": (None, None, "out"),
}


class KDProjectors(eqx.Module):
    """Linear projectors that map student feature dims → teacher feature dims."""

    kv_cond_proj: Optional[eqx.nn.Linear]
    sa_projs: list
    ca_projs: list

    def __init__(self, student_dims: dict, teacher_dims: dict, key):
        kv_key, sa_key, ca_key = jr.split(key, 3)

        s_kv = student_dims["kv_cond"]
        t_kv = teacher_dims["kv_cond"]
        self.kv_cond_proj = (
            None
            if s_kv == t_kv
            else eqx.nn.Linear(s_kv, t_kv, use_bias=False, key=kv_key)
        )

        sa_keys = jr.split(sa_key, len(student_dims["sa"]))
        self.sa_projs = [
            None if s == t else eqx.nn.Linear(s, t, use_bias=False, key=k)
            for (s, t, k) in zip(student_dims["sa"], teacher_dims["sa"], sa_keys)
        ]

        ca_keys = jr.split(ca_key, len(student_dims["ca"]))
        self.ca_projs = [
            None if s == t else eqx.nn.Linear(s, t, use_bias=False, key=k)
            for (s, t, k) in zip(student_dims["ca"], teacher_dims["ca"], ca_keys)
        ]

    def project_kv_cond(self, feat: jnp.ndarray) -> jnp.ndarray:
        if self.kv_cond_proj is None:
            return feat
        return jax.vmap(self.kv_cond_proj)(feat)

    def project_sa(self, layer_idx: int, feat: jnp.ndarray) -> jnp.ndarray:
        proj = self.sa_projs[layer_idx]
        if proj is None:
            return feat
        return jax.vmap(proj)(feat)

    def project_ca(self, layer_idx: int, feat: jnp.ndarray) -> jnp.ndarray:
        proj = self.ca_projs[layer_idx]
        if proj is None:
            return feat
        return jax.vmap(proj)(feat)


class KDLoss(eqx.Module):
    """Knowledge-distillation loss; pass teacher and projectors at call time."""

    accel_scale: float
    yaw_rate_scale: float
    lambdas: dict
    distill_features: frozenset[str]

    def __init__(
        self,
        lambdas: dict,
        distill_features: Iterable[str],
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ):
        unknown = set(distill_features) - DISTILL_FEATURE_SPECS.keys()
        if unknown:
            raise ValueError(f"Unknown distill_features: {sorted(unknown)}")
        self.lambdas = lambdas
        self.distill_features = frozenset(distill_features)
        self.accel_scale = accel_scale
        self.yaw_rate_scale = yaw_rate_scale

    def __call__(
        self,
        model,
        diffusion_sampler,
        agent_past,
        agent_future,
        agents_coeffs,
        agent_future_valid,
        actions_future,
        key,
        teacher=None,
        projectors: KDProjectors | None = None,
        debug=False,
        **kwargs,
    ):
        gt_actions = jnp.asarray(actions_future, dtype=jnp.float32)
        action_scale = jnp.asarray(
            [self.accel_scale, self.yaw_rate_scale], dtype=gt_actions.dtype
        )
        gt_actions_norm = gt_actions / action_scale

        timestep_key, noise_key = jr.split(key)
        timestep = jr.randint(
            timestep_key, shape=(), minval=0, maxval=diffusion_sampler.num_steps
        )
        noise = jr.normal(noise_key, gt_actions.shape)
        noisy_actions = diffusion_sampler.add_noise(gt_actions_norm, noise, timestep)

        student_out, student_feats = model.__call_with_features__(
            timestep, noisy_actions, **kwargs
        )

        teacher_out = None
        teacher_feats = None
        if teacher is not None:
            teacher_out, teacher_feats = teacher.__call_with_features__(
                timestep, noisy_actions, **kwargs
            )
            teacher_out = jax.lax.stop_gradient(teacher_out)
            teacher_feats = jax.lax.stop_gradient(teacher_feats)

        pred_xy, _ = predictions_to_local_xy(
            student_out,
            agent_past=agent_past,
            origin_vel=kwargs["origin_vel"],
            agent_future=agent_future,
            actions_future=actions_future,
            accel_scale=self.accel_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        gt_xy = jnp.asarray(agent_future[..., :2], dtype=jnp.float32)
        err = (pred_xy - gt_xy) ** 2
        valid_target = agent_future_valid
        if valid_target.ndim == err.ndim - 1:
            valid_target = valid_target[..., None]
        weights = jnp.asarray(agents_coeffs, dtype=err.dtype)[..., None, None]
        weights = weights * jnp.asarray(valid_target, dtype=err.dtype)
        weights = jnp.broadcast_to(weights, err.shape)
        l_gt = (err * weights).sum() / jnp.maximum(weights.sum(), 1.0)

        stats = {"L_gt": l_gt}
        total = self.lambdas["gt"] * l_gt

        if teacher is None:
            if debug:
                return total, stats
            return total

        if projectors is None:
            raise ValueError(
                "projectors are required when teacher is set for distillation"
            )

        for feat_key in self.distill_features:
            tensor_name, proj_method, lam_key = DISTILL_FEATURE_SPECS[feat_key]
            lam = self.lambdas.get(lam_key, 0.0)
            if lam == 0.0:
                continue

            if feat_key == "out":
                loss_term = jnp.mean((student_out - teacher_out) ** 2)
            elif feat_key == "kv_cond":
                s_feat = projectors.project_kv_cond(student_feats[tensor_name])
                loss_term = jnp.mean((s_feat - teacher_feats[tensor_name]) ** 2)
            else:
                t_list = teacher_feats[tensor_name]
                s_list = student_feats[tensor_name]
                if len(t_list) == 0:
                    loss_term = jnp.zeros(())
                else:
                    project = getattr(projectors, proj_method)
                    layer_losses = [
                        jnp.mean((project(i, s_list[i]) - t_list[i]) ** 2)
                        for i in range(len(t_list))
                    ]
                    loss_term = sum(layer_losses) / len(layer_losses)

            stats[f"L_{lam_key}"] = loss_term
            total = total + lam * loss_term

        if debug:
            return total, stats
        return total
