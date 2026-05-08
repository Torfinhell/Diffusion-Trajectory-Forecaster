import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


from typing import Literal

import jax.nn as jnn
from einops import rearrange


class MapEncoder(eqx.Module):
    point_in: eqx.nn.Linear
    point_out: eqx.nn.Linear
    traffic_light_embed: eqx.nn.Embedding
    type_embed: eqx.nn.Embedding

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128, key=None):
        point_in_key, point_out_key, tl_key, type_key = jr.split(key, 4)
        self.point_in = eqx.nn.Linear(3, hidden_dim, key=point_in_key)
        self.point_out = eqx.nn.Linear(hidden_dim, embed_dim, key=point_out_key)
        self.traffic_light_embed = eqx.nn.Embedding(8, embed_dim, key=tl_key)
        self.type_embed = eqx.nn.Embedding(21, embed_dim, key=type_key)

    def __call__(self, inputs):
        if inputs.ndim == 4:
            if inputs.shape[0] != 1:
                raise ValueError(
                    f"MapEncoder expected a single-sample batch or unbatched input, got {inputs.shape}"
                )
            inputs = inputs[0]
        elif inputs.ndim != 3:
            raise ValueError(f"MapEncoder expected 3D or 4D input, got {inputs.shape}")

        point_features = jax.vmap(
            jax.vmap(lambda point: self.point_out(jnn.relu(self.point_in(point))))
        )(inputs[..., :3])
        output = jnp.max(point_features, axis=-2)
        traffic_light_type = jnp.clip(inputs[:, 0, 3].astype(jnp.int32), 0, 7)
        traffic_light_embed = jax.vmap(self.traffic_light_embed)(traffic_light_type)
        polyline_type = jnp.clip(inputs[:, 0, 4].astype(jnp.int32), 0, 20)
        type_embed = jax.vmap(self.type_embed)(polyline_type)
        return output + traffic_light_embed + type_embed


class TrafficLightEncoder(eqx.Module):
    type_embed: eqx.nn.Embedding

    def __init__(self, embed_dim: int = 256, key=None):
        self.type_embed = eqx.nn.Embedding(8, embed_dim, key=key)

    def __call__(self, inputs):
        if inputs.ndim == 3:
            if inputs.shape[0] != 1:
                raise ValueError(
                    "TrafficLightEncoder expected a single-sample batch or unbatched "
                    f"input, got {inputs.shape}"
                )
            inputs = inputs[0]
        elif inputs.ndim != 2:
            raise ValueError(
                f"TrafficLightEncoder expected 2D or 3D input, got {inputs.shape}"
            )

        traffic_light_type = jnp.clip(inputs[:, 2].astype(jnp.int32), 0, 7)
        return jax.vmap(self.type_embed)(traffic_light_type)

class AttentionMLP(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    dropout_key: jax.random.PRNGKey
    type_attn: Literal["cross", "self"]
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        attn_dim: int,
        attn_num_heads: int,
        out_dim: int,
        mlp_dim: int,
        num_mlp_layers: int,
        drop_attn: float,
        type_attn: Literal["cross", "self"],
        key,
        kv_dim=None,
    ):
        attn_key, mlp_key, self.dropout_key = jr.split(key, 3)
        assert (
            attn_dim % attn_num_heads == 0
        ), "input attn_dim should be divisable by num_heads"
        self.type_attn = type_attn
        if self.type_attn == "self":
            self.attn = eqx.nn.MultiheadAttention(
                num_heads=attn_num_heads,
                query_size=attn_dim,
                dropout_p=drop_attn,
                key=attn_key,
            )
        else:
            self.attn = eqx.nn.MultiheadAttention(
                num_heads=attn_num_heads,
                query_size=attn_dim,
                key_size=kv_dim,
                value_size=kv_dim,
                dropout_p=drop_attn,
                key=attn_key,
            )
        self.mlp = eqx.nn.MLP(
            in_size=attn_dim,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=out_dim,
            activation=jnn.gelu,
            key=mlp_key,
        )
        self.norm1 = eqx.nn.LayerNorm(shape=attn_dim)
        self.norm2 = eqx.nn.LayerNorm(shape=attn_dim)

    def __call__(self, x, kv_cond=None, attn_mask=None):
        # post-norm: residual add first, then normalize (VBD CrossTransformer style)
        if self.type_attn == "self":
            x = jax.vmap(self.norm1)(x + self.attn(x, x, x, mask=attn_mask, key=self.dropout_key))
        else:
            x = jax.vmap(self.norm1)(x + self.attn(x, kv_cond, kv_cond, mask=attn_mask, key=self.dropout_key))
        return jax.vmap(self.norm2)(x + jax.vmap(self.mlp)(x))


class TransformerEncoder(eqx.Module):
    layers: list[AttentionMLP]

    def __init__(
        self,
        layers: int,
        attn_dim: int,
        attn_num_heads: int,
        mlp_dim: int,
        num_mlp_layers: int,
        drop_attn: float,
        key,
    ):
        layer_keys = jr.split(key, layers)
        self.layers = [
            AttentionMLP(
                attn_dim=attn_dim,
                attn_num_heads=attn_num_heads,
                out_dim=attn_dim,
                mlp_dim=mlp_dim,
                num_mlp_layers=num_mlp_layers,
                drop_attn=drop_attn,
                type_attn="self",
                key=layer_key,
            )
            for layer_key in layer_keys
        ]

    def __call__(self, context_tokens, context_mask):
        valid_context = ~context_mask
        self_attn_mask = valid_context[:, None] & valid_context[None, :]
        tokens = jnp.where(context_mask[:, None], 0.0, context_tokens)
        for layer in self.layers:
            tokens = layer(tokens, attn_mask=self_attn_mask)
            tokens = jnp.where(context_mask[:, None], 0.0, tokens)
        return tokens


class FourierEmbedding(eqx.Module):
    freqs: eqx.nn.Embedding
    embed_dim: int

    def __init__(self, embed_dim, key):
        #self.freqs = eqx.nn.Embedding(1, embed_dim // 2, key=key)
        half = embed_dim // 2
        self.freqs = jnp.exp(
            jnp.arange(half) * -(jnp.log(10000.0) / (half - 1))
        )
        self.embed_dim = embed_dim

    def __call__(self, x):
        # return jnp.concatenate(
        #     [jnp.cos(self.freqs.weight * x), jnp.sin(self.freqs.weight * x)], axis=-1
        # ).squeeze(0)[: self.embed_dim]
        args = x * self.freqs
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)])


class RelationEncoder(eqx.Module):
    proj: eqx.nn.MLP

    def __init__(self, hidden_dim: int = 256, key=None):
        self.proj = eqx.nn.MLP(
            in_size=4,
            width_size=hidden_dim,
            depth=1,
            out_size=hidden_dim,
            key=key,
        )

    def __call__(self, relations, pair_mask):
        dx = relations[..., 0]
        dy = relations[..., 1]
        dtheta = relations[..., 2]
        rel_features = jnp.stack(
            [dx, dy, jnp.sin(dtheta), jnp.cos(dtheta)],
            axis=-1,
        )
        edge_emb = jax.vmap(jax.vmap(self.proj))(rel_features)
        edge_emb = jnp.where(pair_mask[..., None], edge_emb, 0.0)
        denom = jnp.maximum(pair_mask.sum(axis=-1, keepdims=True), 1)
        return edge_emb.sum(axis=1) / denom


class SceneEncoder(eqx.Module):
    pos_emb_type: Literal["rope", "lookup", "None"]
    rnn_time: eqx.nn.MultiheadAttention | eqx.nn.LSTMCell
    sa_agents: eqx.nn.MultiheadAttention
    embedding: eqx.nn.Embedding | eqx.nn.RotaryPositionalEmbedding
    mlp: eqx.nn.MLP
    rnn_type: str
    dropout_key: jax.random.PRNGKey

    def __init__(
        self,
        rnn_type: Literal["lstm", "mhsa"],
        rnn_num_heads: int,
        sa_num_heads: int,
        drop_attn: float,
        mlp_dim: int,
        num_mlp_layers: int,
        num_agents: int,
        time_len: int,
        num_feat: int,
        out_dim: int,
        pos_emb_type: Literal["rope", "lookup", "None"],
        rope_theta: float,
        key,
    ):
        rnn_key, sa_key, mlp_key, self.dropout_key, embed_key = jr.split(key, 5)
        self.rnn_type = rnn_type
        in_dim = num_agents * num_feat
        rnn_dim = in_dim
        assert (
            rnn_dim % rnn_num_heads == 0 or rnn_type == "lstm"
        ), "input rnn_dim should be divisable by rnn_num_heads"
        self.rnn_time = (
            eqx.nn.LSTMCell(input_size=in_dim, hidden_size=rnn_dim, key=rnn_key)
            if rnn_type == "lstm"
            else eqx.nn.MultiheadAttention(
                num_heads=rnn_num_heads,
                query_size=rnn_dim,
                dropout_p=drop_attn,
                key=rnn_key,
            )
        )
        sa_dim = time_len * num_feat
        assert (
            sa_dim % sa_num_heads == 0
        ), "input sa_dim should be divisable by num_heads"
        self.sa_agents = eqx.nn.MultiheadAttention(
            num_heads=sa_num_heads, query_size=sa_dim, dropout_p=drop_attn, key=sa_key
        )
        self.mlp = eqx.nn.MLP(
            in_size=time_len * num_feat,
            width_size=mlp_dim,
            depth=max(num_mlp_layers - 1, 0),
            out_size=out_dim,
            key=mlp_key,
        )
        self.pos_emb_type = pos_emb_type
        embedding_size = num_agents * num_feat
        if self.pos_emb_type == "rope":
            self.embedding = eqx.nn.RotaryPositionalEmbedding(
                embedding_size=embedding_size, theta=rope_theta
            )
        elif self.pos_emb_type == "lookup":
            self.embedding = eqx.nn.Embedding(
                num_embeddings=time_len, embedding_size=embedding_size, key=embed_key
            )
        else:
            self.embedding = None

    def __call__(self, x):
        if x.ndim == 3:
            x = x[None, ...]
        elif x.ndim != 4:
            raise ValueError(f"SceneEncoder expected 3D or 4D input, got {x.shape}")
        _, a, t, f = x.shape
        expected_in_dim = (
            self.rnn_time.input_size
            if isinstance(self.rnn_time, eqx.nn.LSTMCell)
            else self.rnn_time.query_size
        )
        actual_in_dim = a * f
        if actual_in_dim != expected_in_dim:
            raise ValueError(
                "SceneEncoder input shape mismatch: expected A*F="
                f"{expected_in_dim} from model config, got A*F={actual_in_dim} "
                f"(A={a}, F={f}). Align model.num_agents/num_feat with dataset preprocessing."
            )
        x = rearrange(x, "1 a t f -> t (a f)")
        if isinstance(self.rnn_time, eqx.nn.LSTMCell):
            assert not isinstance(self.embedding, eqx.nn.RotaryPositionalEmbedding)

            def scan_fn(state, xt):
                new_state = self.rnn_time(xt, state)
                return new_state, new_state[0]

            init_state = (
                jnp.zeros((self.rnn_time.hidden_size,)),
                jnp.zeros((self.rnn_time.hidden_size,)),
            )
            _, x = jax.lax.scan(scan_fn, init_state, x)
            if isinstance(self.embedding, eqx.nn.Embedding):
                x += self.embedding(jnp.arange(0, t))
        else:
            if isinstance(self.embedding, eqx.nn.RotaryPositionalEmbedding):

                def process_heads(query_heads, key_heads, value_heads):
                    query_heads = jax.vmap(self.embedding, in_axes=1, out_axes=1)(
                        query_heads
                    )
                    key_heads = jax.vmap(self.embedding, in_axes=1, out_axes=1)(
                        key_heads
                    )
                    return query_heads, key_heads, value_heads

            else:
                process_heads = None
            x = self.rnn_time(
                x, x, x, key=self.dropout_key, process_heads=process_heads
            )
            if isinstance(self.embedding, eqx.nn.Embedding):
                x += jax.vmap(self.embedding)(jnp.arange(0, t))
        x = rearrange(x, "t (a f) -> a (t f)", a=a)
        x = self.sa_agents(x, x, x, key=self.dropout_key)
        return jax.vmap(self.mlp)(x).reshape(a, -1)


class SimpleAgentEncoder(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, time_len: int, num_feat: int, out_dim: int, key):
        self.mlp = eqx.nn.MLP(
            in_size=time_len * num_feat,
            width_size=out_dim,
            depth=1,
            out_size=out_dim,
            activation=jnn.relu,
            key=key,
        )

    def __call__(self, x):
        if x.ndim == 4:
            x = x[0]  # (a, t, f)
        a = x.shape[0]
        return jax.vmap(self.mlp)(x.reshape(a, -1))


class Encoder(eqx.Module):
    agent_encoder: SceneEncoder | SimpleAgentEncoder
    map_encoder: MapEncoder
    traffic_light_encoder: TrafficLightEncoder
    relation_encoder: RelationEncoder
    relation_proj: eqx.nn.Linear
    transformer_encoder: TransformerEncoder

    def __init__(
        self,
        agent_encoder_args,
        map_embed_dim: int | None = None,
        map_hidden_dim: int = 128,
        traffic_light_embed_dim: int | None = None,
        transformer_layers: int = 2,
        transformer_attn_num_heads: int = 1,
        transformer_mlp_dim: int = 64,
        transformer_num_mlp_layers: int = 1,
        transformer_drop_attn: float = 0.1,
        key=None,
    ):
        agent_key, map_key, traffic_key, relation_key, relation_proj_key, transformer_key = jr.split(key, 6)
        context_dim = int(agent_encoder_args["out_dim"])
        map_embed_dim = context_dim if map_embed_dim is None else int(map_embed_dim)
        traffic_light_embed_dim = (
            context_dim if traffic_light_embed_dim is None else int(traffic_light_embed_dim)
        )
        if agent_encoder_args.get("rnn_type") == "simple_mlp":
            self.agent_encoder = SimpleAgentEncoder(
                time_len=int(agent_encoder_args["time_len"]),
                num_feat=int(agent_encoder_args["num_feat"]),
                out_dim=context_dim,
                key=agent_key,
            )
        else:
            self.agent_encoder = SceneEncoder(**agent_encoder_args, key=agent_key)
        self.map_encoder = MapEncoder(map_embed_dim, map_hidden_dim, key=map_key)
        self.traffic_light_encoder = TrafficLightEncoder(traffic_light_embed_dim, key=traffic_key)
        self.relation_encoder = RelationEncoder(hidden_dim=context_dim, key=relation_key)
        self.relation_proj = eqx.nn.Linear(
            in_features=context_dim,
            out_features=context_dim,
            key=relation_proj_key,
        )
        self.relation_proj = eqx.tree_at(
            lambda layer: layer.weight,
            self.relation_proj,
            jnp.zeros_like(self.relation_proj.weight),
        )
        self.relation_proj = eqx.tree_at(
            lambda layer: layer.bias,
            self.relation_proj,
            jnp.zeros_like(self.relation_proj.bias),
        )
        self.transformer_encoder = TransformerEncoder(
            layers=transformer_layers,
            attn_dim=context_dim,
            attn_num_heads=transformer_attn_num_heads,
            mlp_dim=transformer_mlp_dim,
            num_mlp_layers=transformer_num_mlp_layers,
            drop_attn=transformer_drop_attn,
            key=transformer_key,
        )

    def __call__(
        self, 
        agent_past, 
        polylines, #local
        polylines_valid,
        traffic_light_points,
        relations,
        agents_valid,
        agents_types,
        **kwargs,
        ):
        encoded_agents = self.agent_encoder(agent_past)
        encoded_map_lanes = self.map_encoder(polylines)
        encoded_traffic_lights = self.traffic_light_encoder(traffic_light_points)

        agents_mask = ~agents_valid
        maps_mask = polylines_valid <= 0
        traffic_lights_mask = jnp.all(traffic_light_points == 0, axis=-1)
        context_mask = jnp.concatenate([agents_mask, maps_mask, traffic_lights_mask], axis=0)
        context_tokens = jnp.concatenate(
            [encoded_agents, encoded_map_lanes, encoded_traffic_lights], axis=0
        )
        pair_valid = (~context_mask)[:, None] & (~context_mask)[None, :]
        relation_context = self.relation_encoder(relations, pair_valid)
        relation_context = jax.vmap(self.relation_proj)(relation_context)
        context_tokens = jnp.where(
            context_mask[:, None], 0.0, context_tokens + relation_context
        )
        encodings = self.transformer_encoder(
            context_tokens,
            context_mask,
        )

        outputs = {
            "agents_mask": agents_mask,
            "maps_mask": maps_mask,
            "traffic_lights_mask": traffic_lights_mask,
            "context_mask": context_mask,
            "encodings": encodings,
        }
        if agents_types is not None:
            outputs["agents_types"] = agents_types
        return outputs



class DiffAttention(eqx.Module):
    encoder: Encoder
    out_shape: tuple[int, ...]
    ca_mlp_layers: list[AttentionMLP]
    sa_mlp_layers: list[AttentionMLP]
    noise_level_embedding: eqx.nn.Embedding
    embed_past: FourierEmbedding
    debug_mlp: eqx.nn.MLP
    mlp_out: eqx.nn.Linear
    old_version: bool
    old_masking: bool
    debug_mlp_only: bool
    input_residual: bool
    input_proj: eqx.nn.Sequential

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
        old_version: bool = True,
        old_masking: bool = True,
        debug_mlp_only: bool = False,
        debug_mlp_dim: int | None = None,
        debug_mlp_layers: int = 2,
        input_residual: bool = False,
    ):
        se_key, ca_mlp_key, sa_mlp_key, out_key, future_key, past_key, debug_mlp_key, proj_key = jr.split(key, 8)
        self.encoder = Encoder(
            agent_encoder_args=se_args,
            map_embed_dim=se_args["out_dim"],
            traffic_light_embed_dim=se_args["out_dim"],
            transformer_layers=2,
            transformer_attn_num_heads=camlp_args["attn_num_heads"],
            transformer_mlp_dim=camlp_args["mlp_dim"],
            transformer_num_mlp_layers=camlp_args["num_mlp_layers"],
            transformer_drop_attn=camlp_args["drop_attn"],
            key=se_key,
        )
        t_emb_dim = camlp_args["out_dim"]
        sa_keys = jr.split(sa_mlp_key, num_sa_mlp)
        self.sa_mlp_layers = [
            AttentionMLP(**samlp_args, key=layer_key, type_attn="self")
            for layer_key in sa_keys
        ]
        ca_keys = jr.split(ca_mlp_key, num_camlp)
        self.ca_mlp_layers = [
            AttentionMLP(**camlp_args, key=layer_key, type_attn="cross")
            for layer_key in ca_keys
        ]
        # learned integer timestep embedding, same as VBD noise_level_embedding
        self.noise_level_embedding = eqx.nn.Embedding(num_diffusion_steps, t_emb_dim, key=future_key)
        self.embed_past = FourierEmbedding(se_args["out_dim"], key=past_key)
        debug_width = (
            camlp_args["out_dim"] if debug_mlp_dim is None else int(debug_mlp_dim)
        )
        self.debug_mlp = eqx.nn.MLP(
            in_size=camlp_args["out_dim"],
            width_size=debug_width,
            depth=max(debug_mlp_layers - 1, 0),
            out_size=camlp_args["out_dim"],
            key=debug_mlp_key,
        )
        self.mlp_out = eqx.nn.Linear(
            in_features=camlp_args["out_dim"],
            out_features=final_out_dim,
            key=out_key,
        )
        # 2-layer MLP projection of x_flat → attn_dim (VBD encoder style)
        proj1_key, proj2_key = jr.split(proj_key)
        self.input_proj = eqx.nn.Sequential([
            eqx.nn.Linear(out_shape[1] * 2, t_emb_dim, key=proj1_key),
            eqx.nn.Lambda(jnn.relu),
            eqx.nn.Linear(t_emb_dim, camlp_args["out_dim"], key=proj2_key),
        ])
        self.out_shape = tuple(out_shape)
        self.old_version = bool(old_version)
        self.old_masking = bool(old_masking)
        self.debug_mlp_only = bool(debug_mlp_only)
        self.input_residual = bool(input_residual)

    def __call__(self, t_noise, x_t, batch, old_version=None):
        if x_t.ndim == 3:
            x_t = x_t[None, ...]
        elif x_t.ndim != 4:
            raise ValueError(f"DiffAttention expected x_t with 3 or 4 dims, got {x_t.shape}")

        use_old_version = self.old_version if old_version is None else old_version
        if use_old_version:
            kv_cond = self.encoder.agent_encoder(batch["agent_past"])
            agents_mask = ~batch["agents_valid"]
            context_mask = agents_mask
        else:
            encoder_outputs = self.encoder(**batch)
            kv_cond = encoder_outputs["encodings"]
            context_mask = encoder_outputs["context_mask"]
            agents_mask = encoder_outputs["agents_mask"]

        _, a, _, _ = x_t.shape
        x_t_flat = x_t.reshape(a, -1)  # (a, T*2) — saved for optional input skip
        # recover integer step from normalized float (t_noise = step / (num_steps-1))
        num_steps = self.noise_level_embedding.num_embeddings
        t_int = jnp.round(t_noise * (num_steps - 1)).astype(jnp.int32)
        t_emb = self.noise_level_embedding(t_int)  # (t_emb_dim,)
        x_t = jax.vmap(self.input_proj)(x_t_flat)
        x_t = x_t + t_emb  # broadcast to (a, attn_dim) — VBD: query + noise_level

        def _out(x):
            out = jax.vmap(self.mlp_out)(x)
            if self.input_residual:
                out = out + x_t_flat
            return out.reshape(self.out_shape)

        if self.debug_mlp_only:
            x_t = jax.vmap(self.debug_mlp)(x_t)
            return _out(x_t)

        if use_old_version and self.old_masking:
            diagonal_cross_mask = jnp.diag(jnp.ones((a,), dtype=bool))
            for layer in self.sa_mlp_layers:
                x_t = layer(x_t)
            for layer in self.ca_mlp_layers:
                x_t = layer(x_t, kv_cond, attn_mask=diagonal_cross_mask)
            return _out(x_t)

        valid_agents = ~agents_mask
        valid_context = ~context_mask
        self_attn_mask = valid_agents[:, None] & valid_agents[None, :]
        cross_attn_mask = valid_agents[:, None] & valid_context[None, :]

        x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
        for layer in self.sa_mlp_layers:
            x_t = layer(x_t, attn_mask=self_attn_mask)
            x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
        for layer in self.ca_mlp_layers:
            x_t = layer(x_t, kv_cond, attn_mask=cross_attn_mask)
            x_t = jnp.where(agents_mask[:, None], 0.0, x_t)

        return _out(x_t)
