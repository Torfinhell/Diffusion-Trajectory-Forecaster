"""JAX/Equinox port of the VBD (Versatile Behavior Diffusion) model.

Faithful adaptation of `vbd_model/` (PyTorch reference) to this repo's
conventions:

* **Per-scene** (no batch dim) -- the trainer vmaps over the batch, matching
  the rest of the models here. All shapes below are for a single scene.
* **Action diffusion**: the diffusion variable is per-agent actions
  ``[A, num_actions, 2] = [accel, yaw_rate]`` (NOT raw positions). The denoiser
  predicts clean actions (``prediction_type="sample"``); a differentiable
  kinematic rollout (``src.utils.path_kinematics.roll_out``) turns them into a
  trajectory for the loss. This is VBD's central idea and the reason it is well
  conditioned.
* **Local coords**: the reference transforms agents/polylines to per-agent
  local frames inside the encoder. *Our* data is already stored in local
  coords, so those transforms are skipped -- we consume the encodings the
  existing ``SceneEncoder`` pathway would produce. The rollout for the loss is
  still done in the global frame (as in VBD) via ``roll_out(global_frame=True)``.

Architecture mirrors ``vbd_model/modules.py``:
  RelQCMHA  <- QCMHA            (relation-injected multi-head attention)
  SelfTransformer / CrossTransformer / TransformerEncoder
  Denoiser / TransformerDecoder (per-agent variant; no cross-agent causal mask)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from src.models.encoders.scene import SceneEncoder
from src.utils.path_kinematics import roll_out


def _linear(in_dim, out_dim, key, *, bias=True):
    return eqx.nn.Linear(in_dim, out_dim, use_bias=bias, key=key)


class RelQCMHA(eqx.Module):
    """Relation-injected multi-head attention (port of ``QCMHA``).

    Operates on a single scene: ``query`` is ``[N, D]`` and ``rel_pos`` is
    ``[N, N, D]`` (per query/key pair). Relations are added to both the
    attention logits and the aggregated values, per head.
    """

    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim, num_heads, key):
        assert embed_dim % num_heads == 0, "embed_dim must divide num_heads"
        k1, k2 = jr.split(key)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.in_proj = _linear(embed_dim, 3 * embed_dim, k1)
        self.out_proj = _linear(embed_dim, embed_dim, k2)

    def __call__(self, query, rel_pos=None, attn_mask=None):
        n, d = query.shape
        h, hd = self.num_heads, self.head_dim

        qkv = jax.vmap(self.in_proj)(query)  # [N, 3D]
        qkv = qkv.reshape(n, h, 3 * hd)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [N, h, hd]
        q = q.transpose(1, 0, 2)  # [h, N, hd]
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        logits = jnp.einsum("hqd,hkd->hqk", q, k)  # [h, N, N]

        if rel_pos is not None:
            rp = rel_pos.reshape(n, n, h, hd)
            # logit contribution: <q_i, rel_{i,k}>  -> [h, N(q), N(k)]
            rp_q = rp.transpose(2, 0, 1, 3)  # [h, q, k, hd]
            logits = logits + jnp.einsum("hqd,hqkd->hqk", q, rp_q)

        logits = logits / jnp.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask: [N, N] True == masked (invalid key)
            logits = logits - attn_mask[None].astype(logits.dtype) * 1e9

        attn = jnn.softmax(logits, axis=-1)  # [h, N, N]
        out = jnp.einsum("hqk,hkd->hqd", attn, v)  # [h, N, hd]

        if rel_pos is not None:
            rp_v = rp.transpose(2, 0, 1, 3)  # [h, q, k, hd]
            out = out + jnp.einsum("hqk,hqkd->hqd", attn, rp_v)

        out = out.transpose(1, 0, 2).reshape(n, d)  # [N, D]
        return jax.vmap(self.out_proj)(out)


class _FFN(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim, key, mult=4):
        k1, k2 = jr.split(key)
        self.fc1 = _linear(dim, dim * mult, k1)
        self.fc2 = _linear(dim * mult, dim, k2)

    def __call__(self, x):
        return jax.vmap(self.fc2)(jnn.gelu(jax.vmap(self.fc1)(x)))


class SelfTransformer(eqx.Module):
    """Post-norm self-attention block with relations (port of ``SelfTransformer``)."""

    attn: RelQCMHA
    ffn: _FFN
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, dim, heads, key):
        ka, kf = jr.split(key)
        self.attn = RelQCMHA(dim, heads, ka)
        self.ffn = _FFN(dim, kf)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)

    def __call__(self, x, relations, attn_mask=None):
        a = self.attn(x, relations, attn_mask)
        x = jax.vmap(self.norm1)(a + x)
        f = self.ffn(x)
        return jax.vmap(self.norm2)(f + x)


class CrossTransformer(eqx.Module):
    """Cross-attention block (port of ``CrossTransformer``).

    Query ``[Q, D]`` attends to key/value ``[K, D]`` with relations
    ``[Q, K, D]`` added to keys/values. Standard scaled-dot MHA.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    ffn: _FFN
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(self, dim, heads, key, residual: bool = True):
        kq, kk, kv, ko, kf = jr.split(key, 5)
        assert dim % heads == 0
        self.num_heads = int(heads)
        self.head_dim = dim // heads
        self.residual = bool(residual)
        self.q_proj = _linear(dim, dim, kq)
        self.k_proj = _linear(dim, dim, kk)
        self.v_proj = _linear(dim, dim, kv)
        self.out_proj = _linear(dim, dim, ko)
        self.ffn = _FFN(dim, kf)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)

    def __call__(self, query, key, relations=None, key_mask=None):
        # key, value get relations added (VBD: key = key + relations; value = key)
        if relations is not None:
            kv_in = key[None] + relations  # [Q, K, D]
        else:
            kv_in = jnp.broadcast_to(key[None], (query.shape[0], *key.shape))

        nq, d = query.shape
        h, hd = self.num_heads, self.head_dim

        q = jax.vmap(self.q_proj)(query).reshape(nq, h, hd)  # [Q, h, hd]
        # per-query keys/values: [Q, K, h, hd]
        k = jax.vmap(jax.vmap(self.k_proj))(kv_in).reshape(nq, -1, h, hd)
        v = jax.vmap(jax.vmap(self.v_proj))(kv_in).reshape(nq, -1, h, hd)

        logits = jnp.einsum("qhd,qkhd->hqk", q, k) / jnp.sqrt(self.head_dim)
        if key_mask is not None:
            logits = logits - key_mask[None, None, :].astype(logits.dtype) * 1e9
        attn = jnn.softmax(logits, axis=-1)  # [h, Q, K]
        out = jnp.einsum("hqk,qkhd->qhd", attn, v).reshape(nq, d)
        out = jax.vmap(self.out_proj)(out)

        # Faithful VBD adds the query residual so the noised actions / timestep
        # signal carried by ``query`` actually propagates to the prediction.
        # Without it the block collapses to a context-conditioned mean and the
        # sampling chain drifts (toggle via ``cross_residual``).
        if self.residual:
            out = out + query
        x = jax.vmap(self.norm1)(out)
        f = self.ffn(x)
        return jax.vmap(self.norm2)(f + x)


class DiffVBD(eqx.Module):
    """VBD-style action-diffusion denoiser (per scene).

    ``__call__(t_noise, x_t, *, past_path, **batch)`` matches the trainer
    contract used by the other models. ``x_t`` is the noised action tensor
    ``[A, num_actions, 2]``; the model returns predicted clean actions of the
    same shape (``prediction_type="sample"``).
    """

    encoder: SceneEncoder
    noise_level_embedding: eqx.nn.Embedding
    time_embedding: eqx.nn.Embedding
    action_in: eqx.nn.MLP
    self_layers: list
    cross_layers: list
    out_decoder: eqx.nn.MLP
    causal_mask: jnp.ndarray
    out_shape: tuple
    action_len: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    num_agents: int = eqx.field(static=True)
    cross_agent_causal: bool = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self,
        se_args,
        num_self_layers,
        num_cross_layers,
        num_actions: int,
        action_len: int,
        dim: int,
        heads: int,
        num_diffusion_steps: int,
        key,
        dt: float = 0.1,
        extract_map: bool = True,
        extract_traffic: bool = True,
        extract_relations: bool = True,
        cross_agent_causal: bool = False,
        cross_residual: bool = True,
    ):
        (
            se_key,
            self_key,
            cross_key,
            nl_key,
            t_key,
            in_key,
            out_key,
        ) = jr.split(key, 7)

        se_args = dict(se_args)
        self.num_agents = int(se_args["num_agents"])
        self.cross_agent_causal = bool(cross_agent_causal)
        combiner_type = se_args.pop("combiner_type", "context_combiner")
        combiner_args = se_args.pop("combiner_args", None)
        self.encoder = SceneEncoder(
            agent_encoder_args=se_args,
            map_embed_dim=dim,
            traffic_light_embed_dim=dim,
            context_hidden_dim=dim,
            extract_map=extract_map,
            extract_traffic=extract_traffic,
            extract_relations=extract_relations,
            combiner_type=combiner_type,
            combiner_args=combiner_args,
            key=se_key,
        )

        self.num_actions = int(num_actions)
        self.action_len = int(action_len)
        self.dt = float(dt)
        # Diffusion variable shape (matches DiffAttention's denoise_shape
        # convention): [num_agents, num_actions, 2]. The sampler uses this as
        # the noise shape, so it MUST include the agent dimension.
        self.out_shape = (self.num_agents, self.num_actions, 2)

        self.noise_level_embedding = eqx.nn.Embedding(num_diffusion_steps, dim, key=nl_key)
        self.time_embedding = eqx.nn.Embedding(self.num_actions, dim, key=t_key)
        # per-action-token encoder: [accel, yaw_rate] -> dim
        self.action_in = eqx.nn.MLP(
            in_size=2, width_size=dim, depth=1, out_size=dim, key=in_key
        )
        self.self_layers = [
            SelfTransformer(dim, heads, k)
            for k in jr.split(self_key, num_self_layers)
        ]
        self.cross_layers = [
            CrossTransformer(dim, heads, k, residual=bool(cross_residual))
            for k in jr.split(cross_key, num_cross_layers)
        ]
        self.out_decoder = eqx.nn.MLP(
            in_size=dim, width_size=dim, depth=1, out_size=2, key=out_key
        )
        # Cross-agent causal mask over flattened [A*T] tokens (token order is
        # agent-major: index = agent*T + time). True == masked (cannot attend).
        # An agent attends to all of its OWN action tokens, and to OTHER agents'
        # tokens only up to and including the current timestep (port of VBD's
        # ``TransformerDecoder.generate_casual_mask``).
        a, t = self.num_agents, self.num_actions
        ai = jnp.arange(a * t) // t          # agent id of each token
        ti = jnp.arange(a * t) % t           # timestep of each token
        same_agent = ai[:, None] == ai[None, :]
        allowed = same_agent | (ti[None, :] <= ti[:, None])  # [A*T, A*T]
        self.causal_mask = ~allowed

    def __call__(self, t_noise, x_t, *, past_path, **batch):
        # x_t: [A, num_actions, 2] noised actions (a leading singleton batch
        # dim is tolerated, mirroring DiffAttention).
        if x_t.ndim == 4:
            x_t = x_t[0]
        elif x_t.ndim != 3:
            raise ValueError(
                f"DiffVBD expected x_t with 3 or 4 dims, got {x_t.shape}"
            )
        a, t, _ = x_t.shape

        enc = self.encoder(past_path=past_path, **batch)
        encodings = enc["encodings"]  # [Ctx, D]
        context_mask = enc["context_mask"]  # [Ctx] True == invalid

        # query: per-agent, per-action-token embedding of the noised actions
        action_emb = jax.vmap(jax.vmap(self.action_in))(x_t)  # [A, T, D]
        t_emb = jax.vmap(self.time_embedding)(jnp.arange(t))  # [T, D]
        nl_emb = self.noise_level_embedding(t_noise)  # [D]
        query = action_emb + t_emb[None] + nl_emb[None, None]  # [A, T, D]

        if self.cross_agent_causal:
            # Joint self-attention over all agents' action tokens with the
            # cross-agent causal mask, then per-agent cross-attention to scene.
            x = query.reshape(a * t, -1)  # [A*T, D]
            for layer in self.self_layers:
                x = layer(x, None, self.causal_mask)
            x = x.reshape(a, t, -1)

            def cross_agent(x_agent):  # [T, D]
                for layer in self.cross_layers:
                    x_agent = layer(
                        x_agent, encodings, relations=None, key_mask=context_mask
                    )
                return jax.vmap(self.out_decoder)(x_agent)

            return jax.vmap(cross_agent)(x)  # [A, T, 2]

        # per-agent decode: each agent attends only over its OWN action tokens,
        # then over the scene encodings ("faithful minus cross-agent causal").
        def decode_agent(q_agent):  # q_agent: [T, D]
            x = q_agent
            for layer in self.self_layers:
                x = layer(x, None, None)  # temporal self-attention over own tokens
            for layer in self.cross_layers:
                x = layer(x, encodings, relations=None, key_mask=context_mask)
            return jax.vmap(self.out_decoder)(x)  # [T, 2]

        return jax.vmap(decode_agent)(query)  # [A, T, 2]
