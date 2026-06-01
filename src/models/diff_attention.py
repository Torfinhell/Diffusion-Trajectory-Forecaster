import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from src.models.components.attention import CrossAttentionMLP, SelfAttentionMLP
from src.models.components.fourier import FourierEmbedding
from src.models.encoders.scene import SceneEncoder


class DiffAttention(eqx.Module):
    encoder: SceneEncoder
    out_shape: tuple[int, ...]
    ca_mlp_layers: list[CrossAttentionMLP]
    sa_mlp_layers: list[SelfAttentionMLP]
    embed_past: FourierEmbedding
    noise_level_embedding: eqx.nn.Embedding
    input_proj: eqx.nn.Sequential
    mlp_out: eqx.nn.Linear
    diagonal_ca: bool

    def __init__(
        self,
        se_args,
        samlp_args,
        num_sa_mlp,
        camlp_args,
        num_camlp,
        out_shape: list[int],
        final_out_dim: int,
        key,
        num_diffusion_steps: int = 50,
        extract_map: bool = True,
        extract_traffic: bool = True,
        extract_relations: bool = True,
        diagonal_ca: bool | None = None,
    ):
        se_key, ca_mlp_key, sa_mlp_key, out_key, future_key, past_key, proj_key = (
            jr.split(key, 7)
        )
        se_args = dict(se_args)
        combiner_type = se_args.pop("combiner_type", "context_combiner")
        combiner_args = se_args.pop("combiner_args", None)
        self.encoder = SceneEncoder(
            agent_encoder_args=se_args,
            map_embed_dim=se_args["out_dim"],
            traffic_light_embed_dim=se_args["out_dim"],
            context_hidden_dim=camlp_args["mlp_dim"],
            extract_map=extract_map,
            extract_traffic=extract_traffic,
            extract_relations=extract_relations,
            combiner_type=combiner_type,
            combiner_args=combiner_args,
            key=se_key,
        )
        sa_keys = jr.split(sa_mlp_key, num_sa_mlp)
        self.sa_mlp_layers = [
            SelfAttentionMLP(**samlp_args, key=layer_key) for layer_key in sa_keys
        ]
        ca_keys = jr.split(ca_mlp_key, num_camlp)
        camlp_kwargs = dict(camlp_args)
        kv_dim = camlp_kwargs.pop("kv_dim")
        self.ca_mlp_layers = [
            CrossAttentionMLP(**camlp_kwargs, kv_dim=kv_dim, key=layer_key)
            for layer_key in ca_keys
        ]
        t_emb_dim = camlp_args["out_dim"]
        self.noise_level_embedding = eqx.nn.Embedding(
            num_diffusion_steps, t_emb_dim, key=future_key
        )
        self.embed_past = FourierEmbedding(se_args["out_dim"], key=past_key)
        self.mlp_out = eqx.nn.Linear(
            in_features=camlp_args["out_dim"],
            out_features=final_out_dim,
            key=out_key,
        )
        proj1_key, proj2_key = jr.split(proj_key)
        self.input_proj = eqx.nn.Sequential(
            [
                eqx.nn.Linear(out_shape[1] * 2, t_emb_dim, key=proj1_key),
                eqx.nn.Lambda(jnn.relu),
                eqx.nn.Linear(t_emb_dim, camlp_args["out_dim"], key=proj2_key),
            ]
        )
        self.out_shape = tuple(out_shape)
        self.diagonal_ca = diagonal_ca if diagonal_ca is not None else (combiner_type != "transformer")

    def _forward_impl(self, t_noise, x_t, *, collect_features: bool, **batch_kwargs):
        if x_t.ndim == 3:
            x_t = x_t[None, ...]
        elif x_t.ndim != 4:
            raise ValueError(
                f"DiffAttention expected x_t with 3 or 4 dims, got {x_t.shape}"
            )

        encoder_outputs = self.encoder(**batch_kwargs)
        kv_cond = encoder_outputs["encodings"]
        context_mask = encoder_outputs["context_mask"]
        agents_mask = encoder_outputs["agents_mask"]

        _, a, _, _ = x_t.shape
        x_t_flat = x_t.reshape(a, -1)
        t_emb = self.noise_level_embedding(t_noise)
        x_t = jax.vmap(self.input_proj)(x_t_flat)  # TODO
        x_t = x_t + t_emb

        valid_agents = ~agents_mask
        valid_context = ~context_mask
        self_attn_mask = valid_agents[:, None] & valid_agents[None, :]
        if self.diagonal_ca:
            cross_attn_mask = jnp.diag(valid_agents) & valid_context[None, :]
        else:
            cross_attn_mask = valid_agents[:, None] & valid_context[None, :]

        x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
        sa = [] if collect_features else None
        for layer in self.sa_mlp_layers:
            x_t = layer(x_t, attn_mask=self_attn_mask)
            x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
            if collect_features:
                sa.append(x_t)

        ca = [] if collect_features else None
        for layer in self.ca_mlp_layers:
            x_t = layer(x_t, kv_cond, attn_mask=cross_attn_mask)
            x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
            if collect_features:
                ca.append(x_t)

        out = jax.vmap(self.mlp_out)(x_t).reshape(self.out_shape)
        if not collect_features:
            return out

        return out, {"kv_cond": kv_cond, "sa": sa, "ca": ca}

    def __call__(self, t_noise, x_t, **batch_kwargs):
        return self._forward_impl(t_noise, x_t, collect_features=False, **batch_kwargs)

    def __call_with_features__(
        self, t_noise, x_t, **batch_kwargs
    ) -> tuple[jnp.ndarray, dict]:
        return self._forward_impl(t_noise, x_t, collect_features=True, **batch_kwargs)
