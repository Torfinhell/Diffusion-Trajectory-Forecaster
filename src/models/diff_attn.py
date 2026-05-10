from typing import Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
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
            key=mlp_key,
        )
        self.norm1 = eqx.nn.LayerNorm(shape=attn_dim)
        self.norm2 = eqx.nn.LayerNorm(shape=attn_dim)

    def __call__(self, x, kv_cond=None, attn_mask=None):
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
        half = embed_dim // 2
        self.freqs = jnp.exp(
            jnp.arange(half) * -(jnp.log(10000.0) / (half - 1))
        )

    def __call__(self, x):
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


class AgentEncoder(eqx.Module):
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
    

class ContextCombiner(eqx.Module):
    """Per-agent context encoder. Zero-init additive linears fuse agent + scene tokens.
    Each projection takes cat(agent_enc, scene_token) → out_dim, zero-init weight.
    At init: all projections output 0 → out = agent_enc (exact identity)."""
    map_proj: eqx.nn.Linear  # Linear(agent_dim + map_dim, out_dim), zero-init
    tl_proj:  eqx.nn.Linear  # Linear(agent_dim + tl_dim,  out_dim), zero-init
    rel_proj: eqx.nn.Linear  # Linear(agent_dim + rel_dim, out_dim), zero-init

    def __init__(self, agent_dim: int, out_dim: int, hidden_dim: int, key,
                 map_dim: int = 0, tl_dim: int = 0, rel_dim: int = 0):
        def _zero_linear(in_dim, fold_id):
            lin = eqx.nn.Linear(max(int(in_dim), 1), out_dim, use_bias=False, key=jr.fold_in(key, fold_id))
            return eqx.tree_at(lambda l: l.weight, lin, jnp.zeros_like(lin.weight))

        self.map_proj = _zero_linear(agent_dim + map_dim, 0)
        self.tl_proj  = _zero_linear(agent_dim + tl_dim,  1)
        self.rel_proj = _zero_linear(agent_dim + rel_dim, 2)

    def __call__(self, agent_encodings, agents_mask, scene_map=None, scene_tl=None, scene_rel=None):
        out = agent_encodings  # identity at init
        a = agent_encodings.shape[0]
        if scene_map is not None:
            inp = jnp.concatenate([agent_encodings, jnp.broadcast_to(scene_map[None], (a, scene_map.shape[0]))], axis=-1)
            out = out + jax.vmap(self.map_proj)(inp)
        if scene_tl is not None:
            inp = jnp.concatenate([agent_encodings, jnp.broadcast_to(scene_tl[None], (a, scene_tl.shape[0]))], axis=-1)
            out = out + jax.vmap(self.tl_proj)(inp)
        if scene_rel is not None:
            inp = jnp.concatenate([agent_encodings, scene_rel], axis=-1)
            out = out + jax.vmap(self.rel_proj)(inp)
        return jnp.where(agents_mask[:, None], 0.0, out)


class SceneEncoder(eqx.Module):
    agent_encoder: AgentEncoder | SimpleAgentEncoder
    map_encoder: MapEncoder | None
    traffic_light_encoder: TrafficLightEncoder | None
    relation_encoder: RelationEncoder | None
    context_combiner: ContextCombiner
    extract_map: bool
    extract_traffic: bool
    extract_relations: bool

    def __init__(
        self,
        agent_encoder_args,
        map_embed_dim: int | None = None,
        map_hidden_dim: int = 128,
        traffic_light_embed_dim: int | None = None,
        rel_embed_dim: int | None = None,
        context_hidden_dim: int = 256,
        extract_map: bool = True,
        extract_traffic: bool = True,
        extract_relations: bool = True,
        key=None,
        **kwargs,
    ):
        agent_key, map_key, traffic_key, combiner_key = jr.split(key, 4)
        rel_key = jr.fold_in(key, 42)
        context_dim = int(agent_encoder_args["out_dim"])
        map_embed_dim = context_dim if map_embed_dim is None else int(map_embed_dim)
        traffic_light_embed_dim = (
            context_dim
            if traffic_light_embed_dim is None
            else int(traffic_light_embed_dim)
        )
        rel_embed_dim = context_dim if rel_embed_dim is None else int(rel_embed_dim)
        self.agent_encoder = SimpleAgentEncoder(
                time_len=int(agent_encoder_args["time_len"]),
                num_feat=int(agent_encoder_args["num_feat"]),
                out_dim=context_dim,
                key=agent_key,
            )
        self.extract_map = bool(extract_map)
        self.extract_traffic = bool(extract_traffic)
        self.extract_relations = bool(extract_relations)
        self.map_encoder = (
            MapEncoder(map_embed_dim, map_hidden_dim, key=map_key)
            if self.extract_map
            else None
        )
        self.traffic_light_encoder = (
            TrafficLightEncoder(traffic_light_embed_dim, key=traffic_key)
            if self.extract_traffic
            else None
        )
        self.relation_encoder = (
            RelationEncoder(hidden_dim=rel_embed_dim, key=rel_key)
            if self.extract_relations
            else None
        )
        self.context_combiner = ContextCombiner(
            agent_dim=context_dim,
            out_dim=context_dim,
            hidden_dim=context_hidden_dim,
            key=combiner_key,
            map_dim=map_embed_dim,
            tl_dim=traffic_light_embed_dim,
            rel_dim=rel_embed_dim,
        )

    def __call__(
        self,
        agent_past,
        polylines=None,  # local
        polylines_valid=None,
        traffic_light_points=None,
        relations=None,
        agents_valid=None,
        agents_types=None,
        **kwargs,
    ):
        encoded_agents = self.agent_encoder(agent_past)
        encoded_map_lanes = (
            self.map_encoder(polylines)
            if self.map_encoder is not None and polylines is not None
            else None
        )
        encoded_tl = (
            self.traffic_light_encoder(traffic_light_points)
            if self.traffic_light_encoder is not None and traffic_light_points is not None
            else None
        )

        agents_mask = ~agents_valid
        maps_mask = (
            polylines_valid <= 0
            if polylines_valid is not None
            else jnp.ones((0,), dtype=bool)
        )
        traffic_lights_mask = (
            jnp.all(traffic_light_points == 0, axis=-1)
            if traffic_light_points is not None
            else jnp.ones((0,), dtype=bool)
        )

        # Pool valid map lanes → scene map token
        if encoded_map_lanes is not None:
            valid_lanes = (~maps_mask).astype(jnp.float32)
            scene_map = (encoded_map_lanes * valid_lanes[:, None]).sum(0) / jnp.maximum(valid_lanes.sum(), 1.0)
        else:
            scene_map = None
        # Pool valid traffic lights → scene TL token
        if encoded_tl is not None:
            valid_tl = (~traffic_lights_mask).astype(jnp.float32)
            scene_tl = (encoded_tl * valid_tl[:, None]).sum(0) / jnp.maximum(valid_tl.sum(), 1.0)
        else:
            scene_tl = None
        # Per-agent relation embeddings: mean-pool neighbor edge embeddings
        scene_rel = None
        if self.relation_encoder is not None and relations is not None:
            if relations.ndim == 4:
                relations = relations[0]  # squeeze batch dim
            a = encoded_agents.shape[0]
            agent_relations = relations[:a, :a, :]  # (a, a, 3)
            self_loop = jnp.eye(a, dtype=bool)
            pair_mask = agents_valid[:, None] & agents_valid[None, :] & ~self_loop
            scene_rel = self.relation_encoder(agent_relations, pair_mask)  # (a, rel_dim)
        else:
            scene_rel = None
        # Per-agent context: agent fused with scene map + TL + relations
        encodings = self.context_combiner(encoded_agents, agents_mask, scene_map=scene_map, scene_tl=scene_tl, scene_rel=scene_rel)

        agents_types = kwargs.get("agents_types")
        outputs = {
            "agents_mask": agents_mask,
            "maps_mask": maps_mask,
            "traffic_lights_mask": traffic_lights_mask,
            "context_mask": agents_mask,
            "encodings": encodings,
        }
        if agents_types is not None:
            outputs["agents_types"] = agents_types
        return outputs


class DiffAttention(eqx.Module):
    encoder: SceneEncoder
    out_shape: tuple[int, ...]
    ca_mlp_layers: list[AttentionMLP]
    sa_mlp_layers: list[AttentionMLP]
    embed_past: FourierEmbedding
    noise_level_embedding: eqx.nn.Embedding
    input_proj: eqx.nn.Sequential
    mlp_out: eqx.nn.Linear

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
    ):
        se_key, ca_mlp_key, sa_mlp_key, out_key, future_key, past_key, proj_key = jr.split(key, 7)
        self.encoder = SceneEncoder(
            agent_encoder_args=se_args,
            map_embed_dim=se_args["out_dim"],
            traffic_light_embed_dim=se_args["out_dim"],
            context_hidden_dim=camlp_args["mlp_dim"],
            extract_map=extract_map,
            extract_traffic=extract_traffic,
            extract_relations=extract_relations,
            key=se_key,
        )
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
        t_emb_dim = camlp_args["out_dim"]
        # learned integer timestep embedding, same as VBD noise_level_embedding
        self.noise_level_embedding = eqx.nn.Embedding(num_diffusion_steps, t_emb_dim, key=future_key)
        self.embed_past = FourierEmbedding(se_args["out_dim"], key=past_key)
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

    def __call__(self, t_noise, x_t, **batch_kwargs):
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
        x_t = jax.vmap(self.input_proj)(x_t_flat)
        x_t = x_t + t_emb

        valid_agents = ~agents_mask
        valid_context = ~context_mask
        self_attn_mask = valid_agents[:, None] & valid_agents[None, :]
        cross_attn_mask =  jnp.diag(valid_agents) & valid_context[None, :]

        x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
        for layer in self.sa_mlp_layers:
            x_t = layer(x_t, attn_mask=self_attn_mask)
            x_t = jnp.where(agents_mask[:, None], 0.0, x_t)
        for layer in self.ca_mlp_layers:
            x_t = layer(x_t, kv_cond, attn_mask=cross_attn_mask)
            x_t = jnp.where(agents_mask[:, None], 0.0, x_t)

        return jax.vmap(self.mlp_out)(x_t).reshape(self.out_shape)
