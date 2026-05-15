from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.utils.data_utils import predictions_to_local_xy


class KDProjectors(eqx.Module):
    """Linear projectors that map student feature dims → teacher feature dims.

    Projectors are jointly trained with the student and discarded after distillation.
    """

    kv_cond_proj: Optional[eqx.nn.Linear]
    sa_projs: list
    ca_projs: list

    def __init__(self, student_dims: dict, teacher_dims: dict, key):
        """
        Args:
            student_dims: {"kv_cond": int, "sa": list[int], "ca": list[int]}
            teacher_dims: {"kv_cond": int, "sa": list[int], "ca": list[int]}
        """
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
            None
            if s == t
            else eqx.nn.Linear(s, t, use_bias=False, key=k)
            for (s, t, k) in zip(student_dims["sa"], teacher_dims["sa"], sa_keys)
        ]

        ca_keys = jr.split(ca_key, len(student_dims["ca"]))
        self.ca_projs = [
            None
            if s == t
            else eqx.nn.Linear(s, t, use_bias=False, key=k)
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
    """Knowledge-distillation loss with the same signature as MSELoss."""

    teacher: list   # [DiffAttention]
    projectors: KDProjectors
    accel_scale: float
    yaw_rate_scale: float
    lambdas: dict              # {"gt", "scene", "sa", "ca", "out"}

    def __init__(
        self,
        teacher: eqx.Module,
        projectors: KDProjectors,
        lambdas: dict,
        accel_scale: float = 1.0,
        yaw_rate_scale: float = 0.15,
    ):
        self.teacher = [teacher]
        self.projectors = projectors
        self.accel_scale = accel_scale
        self.yaw_rate_scale = yaw_rate_scale
        self.lambdas = lambdas

    # Called once during __init__ of BaseTrainerDebug to build opt_state
    def get_trainable(self, student_model):
        return eqx.filter(
            (student_model, self.projectors), eqx.is_inexact_array
        )

    def __call__(
        self,
        model,           # student DiffAttention
        diffusion_sampler,
        agent_past,
        agent_future,
        agents_coeffs,
        agent_future_valid,
        actions_future,
        actions_future_valid,
        key,
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

        teacher_out, teacher_feats = self.teacher[0].__call_with_features__(
            timestep, noisy_actions, **kwargs)
        teacher_out = jax.lax.stop_gradient(teacher_out)
        teacher_feats = jax.lax.stop_gradient(teacher_feats)

        # Student forward
        student_out, student_feats = model.__call_with_features__(
            timestep, noisy_actions, **kwargs)

        #gt trajectory loss 
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

        # kv_cond loss
        s_kv = self.projectors.project_kv_cond(student_feats["kv_cond"])
        l_scene = jnp.mean((s_kv - teacher_feats["kv_cond"]) ** 2)

        # sa feature loss 
        t_sa = teacher_feats["sa_features"]
        s_sa = student_feats["sa_features"]
        if len(t_sa) > 0:
            sa_losses = [
                jnp.mean(
                    (self.projectors.project_sa(i, s_sa[i]) - t_sa[i]) ** 2
                )
                for i in range(len(t_sa))
            ]
            l_sa = sum(sa_losses) / len(sa_losses)
        else:
            l_sa = jnp.zeros(())

        # ca feature loss 
        t_ca = teacher_feats["ca_features"]
        s_ca = student_feats["ca_features"]
        if len(t_ca) > 0:
            ca_losses = [
                jnp.mean(
                    (self.projectors.project_ca(i, s_ca[i]) - t_ca[i]) ** 2
                )
                for i in range(len(t_ca))
            ]
            l_ca = sum(ca_losses) / len(ca_losses)
        else:
            l_ca = jnp.zeros(())

        # Output loss
        l_out = jnp.mean((student_out - teacher_out) ** 2)

        lam = self.lambdas
        total = (
            lam["gt"] * l_gt
            + lam["scene"] * l_scene
            + lam["sa"] * l_sa
            + lam["ca"] * l_ca
            + lam["out"] * l_out
        )

        if debug:
            stats = {
                "L_gt": l_gt,
                "L_scene": l_scene,
                "L_sa": l_sa,
                "L_ca": l_ca,
                "L_out": l_out,
            }
            return total, stats
        return total
