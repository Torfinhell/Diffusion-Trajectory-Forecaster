import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from src.data_module.agent_path import AgentPath
from src.models.components.context_combiner import ContextCombiner
from src.models.components.map import MapEncoder
from src.models.components.relations import RelationEncoder
from src.models.components.traffic_light import TrafficLightEncoder
from src.models.components.transformer_combiner import TransformerContextCombiner
from src.models.encoders.agent import build_agent_encoder

_COMBINERS = {
    "context_combiner": ContextCombiner,
    "transformer": TransformerContextCombiner,
}


class SceneEncoder(eqx.Module):
    agent_encoder: eqx.Module
    type_embedding: eqx.nn.Embedding
    map_encoder: MapEncoder | None
    traffic_light_encoder: TrafficLightEncoder | None
    relation_encoder: RelationEncoder | None
    context_combiner: eqx.Module

    def __init__(
        self,
        agent_encoder_args,
        map_embed_dim: int | None = None,
        map_hidden_dim: int = 128,
        traffic_light_embed_dim: int | None = None,
        rel_embed_dim: int | None = None,
        context_hidden_dim: int = 256,
        num_agent_types: int = 16,
        extract_map: bool = True,
        extract_traffic: bool = True,
        extract_relations: bool = True,
        combiner_type: str = "context_combiner",
        combiner_args: dict | None = None,
        key=None,
        **kwargs,
    ):
        del kwargs
        agent_key, type_key, map_key, traffic_key, combiner_key = jr.split(key, 5)
        rel_key = jr.fold_in(key, 42)

        args = dict(agent_encoder_args)
        encoder_name = str(args.pop("encoder", "simple"))
        context_dim = int(args["out_dim"])

        self.agent_encoder = build_agent_encoder(encoder_name, key=agent_key, **args)
        self.type_embedding = eqx.nn.Embedding(
            num_embeddings=int(num_agent_types),
            embedding_size=context_dim,
            key=type_key,
        )

        map_embed_dim = context_dim if map_embed_dim is None else int(map_embed_dim)
        traffic_light_embed_dim = (
            context_dim
            if traffic_light_embed_dim is None
            else int(traffic_light_embed_dim)
        )
        rel_embed_dim = context_dim if rel_embed_dim is None else int(rel_embed_dim)

        self.map_encoder = (
            MapEncoder(map_embed_dim, map_hidden_dim, key=map_key)
            if extract_map
            else None
        )
        self.traffic_light_encoder = (
            TrafficLightEncoder(traffic_light_embed_dim, key=traffic_key)
            if extract_traffic
            else None
        )
        self.relation_encoder = (
            RelationEncoder(hidden_dim=rel_embed_dim, key=rel_key)
            if extract_relations
            else None
        )

        combiner_cls = _COMBINERS.get(combiner_type, ContextCombiner)
        extra = dict(combiner_args or {})
        self.context_combiner = combiner_cls(
            agent_dim=context_dim,
            out_dim=context_dim,
            hidden_dim=context_hidden_dim,
            key=combiner_key,
            map_dim=map_embed_dim if extract_map else 0,
            tl_dim=traffic_light_embed_dim if extract_traffic else 0,
            rel_dim=rel_embed_dim if extract_relations else 0,
            **extra,
        )

    def __call__(
        self,
        past_path: AgentPath,
        polylines=None,
        polylines_valid=None,
        traffic_light_points=None,
        relations=None,
        agents_valid=None,
        agents_types=None,
        **kwargs,
    ):
        past_actions, _ = past_path.actions()
        encoded_agents = self.agent_encoder(past_actions)
        if agents_types is not None:
            type_ids = jnp.asarray(agents_types, dtype=jnp.int32)
            type_ids = jnp.clip(type_ids, 0, self.type_embedding.num_embeddings - 1)
            encoded_agents = encoded_agents + jax.vmap(self.type_embedding)(type_ids)

        encoded_map_lanes = (
            self.map_encoder(polylines)
            if self.map_encoder is not None and polylines is not None
            else None
        )
        encoded_tl = (
            self.traffic_light_encoder(traffic_light_points)
            if self.traffic_light_encoder is not None
            and traffic_light_points is not None
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

        scene_map = None
        if encoded_map_lanes is not None:
            valid_lanes = (~maps_mask).astype(jnp.float32)
            scene_map = (encoded_map_lanes * valid_lanes[:, None]).sum(0) / jnp.maximum(
                valid_lanes.sum(), 1.0
            )

        scene_tl = None
        if encoded_tl is not None:
            valid_tl = (~traffic_lights_mask).astype(jnp.float32)
            scene_tl = (encoded_tl * valid_tl[:, None]).sum(0) / jnp.maximum(
                valid_tl.sum(), 1.0
            )

        scene_rel = None
        if self.relation_encoder is not None and relations is not None:
            if relations.ndim == 4:
                relations = relations[0]
            a = encoded_agents.shape[0]
            self_loop = jnp.eye(a, dtype=bool)
            pair_mask = agents_valid[:, None] & agents_valid[None, :] & ~self_loop
            scene_rel = self.relation_encoder(relations[:a, :a], pair_mask)

        result = self.context_combiner(
            encoded_agents,
            agents_mask,
            scene_map=scene_map,
            scene_tl=scene_tl,
            scene_rel=scene_rel,
        )
        if isinstance(result, tuple):
            encodings, context_mask = result
        else:
            encodings, context_mask = result, agents_mask

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
