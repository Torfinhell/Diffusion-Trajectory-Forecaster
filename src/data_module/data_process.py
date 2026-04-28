import jax
import jax.numpy as jnp
import tensorflow as tf
from waymax import dataloader, datatypes

COORD_SCALE = 1.0


@jax.jit(static_argnames=["topk"])
def filter_topk_roadgraph_points(roadgraph, reference_points, topk):
    """
    Returns the topk closest roadgraph points to a reference point.

    If `topk` is larger than the number of points, an exception will be raised.

    Args:
        roadgraph: Roadgraph information to filter, (..., num_points).
        reference_points: A tensor of shape (..., 2) - the reference point used to measure distance.
        topk: Number of points to keep.

    Returns:
        Roadgraph data structure that has been filtered to only contain the `topk` closest points to a reference point.
    """
    num_points = roadgraph.x.shape[-1]
    if topk > num_points:
        raise NotImplementedError("Not enough points in roadgraph.")

    if topk < num_points:
        roadgraph_xy = roadgraph.xy
        dist = jnp.sum((reference_points[..., None, :] - roadgraph_xy) ** 2, axis=-1)
        valid_dist = jnp.where(roadgraph.valid, dist, jnp.inf)
        top_idx = jnp.argpartition(valid_dist, topk, axis=-1)[..., :topk]

        def take(field):
            return jnp.take_along_axis(field, top_idx, axis=-1)

        return datatypes.RoadgraphPoints(
            x=take(roadgraph.x),
            y=take(roadgraph.y),
            z=take(roadgraph.z),
            dir_x=take(roadgraph.dir_x),
            dir_y=take(roadgraph.dir_y),
            dir_z=take(roadgraph.dir_z),
            types=take(roadgraph.types),
            ids=take(roadgraph.ids),
            valid=take(roadgraph.valid),
        )

    else:
        return roadgraph


@jax.jit(static_argnames=["current_index"])
def data_process_traffic_light(scenarios, current_index=10):
    traffic_lights = scenarios.log_traffic_light
    traffic_lane_ids = traffic_lights.lane_ids[..., current_index]
    traffic_light_states = traffic_lights.state[..., current_index]
    traffic_stop_points = traffic_lights.xy[..., current_index, :]
    traffic_light_valid = traffic_lights.valid[..., current_index]

    traffic_light_points = jnp.concatenate(
        [traffic_stop_points, traffic_light_states[..., None]], axis=-1
    )
    traffic_light_points = jnp.float32(traffic_light_points)
    traffic_light_points = jnp.where(
        traffic_light_valid[..., None], traffic_light_points, 0.0
    )

    return {
        "traffic_light_points": traffic_light_points,
        "traffic_lane_ids": traffic_lane_ids,
        "traffic_light_states": traffic_light_states,
    }


@jax.jit(static_argnames=["current_index", "use_full_agent_info"])
def data_process_agent(scenarios, current_index=10, use_full_agent_info=True):
    traj = scenarios.log_trajectory

    if use_full_agent_info:
        agents_info = jnp.stack(
            [
                traj.x,
                traj.y,
                traj.z,
                traj.vel_x,
                traj.vel_y,
                traj.yaw,
                traj.length,
                traj.width,
                traj.height,
            ],
            axis=-1,
        )
    else:
        agents_info = jnp.stack([traj.x, traj.y, traj.z], axis=-1)

    history_valid = traj.valid[..., : current_index + 1]
    history_idx = jnp.arange(current_index + 1, dtype=jnp.int32)
    last_valid_idx = jnp.max(jnp.where(history_valid, history_idx, -1), axis=-1)
    has_history = last_valid_idx >= 0
    safe_last_valid_idx = jnp.maximum(last_valid_idx, 0)
    origin_xyz = jnp.take_along_axis(
        agents_info[..., : current_index + 1, :3],
        safe_last_valid_idx[..., None, None],
        axis=-2,
    ).squeeze(axis=-2)
    origin_xyz = jnp.where(has_history[..., None], origin_xyz, 0.0)

    agents_info = agents_info.at[..., :3].set(
        agents_info[..., :3] - origin_xyz[..., None, :]
    )

    valid_mask = traj.valid[..., None]
    agents_info = jnp.where(valid_mask & has_history[..., None, None], agents_info, 0.0)

    agent_past = agents_info[..., : current_index + 1, :]
    agent_future = agents_info[..., current_index + 1 :, :]
    agent_future_valid = (
        traj.valid[..., current_index + 1 :, None] & has_history[..., None, None]
    )

    is_modeled = scenarios.object_metadata.is_modeled
    is_interesting = scenarios.object_metadata.objects_of_interest
    is_valid = scenarios.object_metadata.is_valid
    agents_coeffs = jnp.where(is_modeled & is_interesting, 10.0, 1.0)
    agents_coeffs = jnp.where(~is_valid, agents_coeffs, 0.0)

    return {
        "agent_past": agent_past,
        "agent_future": agent_future,
        "agent_future_valid": agent_future_valid,
        "agents_coeffs": agents_coeffs,
        "agents_types": scenarios.object_metadata.object_types,
        "origin_xy": origin_xyz[..., :2],
    }

@jax.jit(static_argnames=["max_polylines", "num_points_polyline"])
def data_process_map(
    scenario,
    traffic_info,
    agents_info,
    max_polylines=256,
    num_points_polyline=30,
):
    roadgraph_points = scenario.roadgraph_points
    traffic_light_states = traffic_info["traffic_light_states"]
    traffic_lane_ids = traffic_info["traffic_lane_ids"]
    agents_coeffs = agents_info["agents_coeffs"]
    agent_past = agents_info["agent_past"]
    current_valid = agents_coeffs > 0
    map_ids = []

    for a in range(agent_past.shape[0]):
        agent_position = agent_past[a, -1, :2]
        nearby_roadgraph_points = filter_topk_roadgraph_points(
            roadgraph_points, agent_position, 3000
        )
        agent_ids = jnp.where(current_valid[a], nearby_roadgraph_points.ids, -1)
        map_ids.append(agent_ids)

    map_ids = jnp.stack(map_ids, axis=0)

    sorted_map_ids = jnp.full((max_polylines,), -1, dtype=map_ids.dtype)
    insert_count = jnp.int32(0)
    for i in range(map_ids.shape[1]):
        for j in range(map_ids.shape[0]):
            cur_id = map_ids[j, i]
            already_seen = jnp.any(sorted_map_ids == cur_id)
            can_insert = (cur_id != -1) & (~already_seen) & (insert_count < max_polylines)
            sorted_map_ids = jax.lax.cond(
                can_insert,
                lambda arr: arr.at[insert_count].set(cur_id),
                lambda arr: arr,
                sorted_map_ids,
            )
            insert_count = insert_count + can_insert.astype(jnp.int32)

    roadgraph_points_x = roadgraph_points.x
    roadgraph_points_y = roadgraph_points.y
    roadgraph_points_dir_x = roadgraph_points.dir_x
    roadgraph_points_dir_y = roadgraph_points.dir_y
    roadgraph_points_types = roadgraph_points.types
    roadgraph_heading = jnp.arctan2(roadgraph_points_dir_y, roadgraph_points_dir_x)

    polylines = []
    polylines_valid = []

    for idx in range(sorted_map_ids.shape[0]):
        id = sorted_map_ids[idx]
        point_mask = (roadgraph_points.ids == id) & roadgraph_points.valid
        lane_type = jnp.where(point_mask, roadgraph_points_types, 0)
        traffic_light_state = jnp.max(
            jnp.where(traffic_lane_ids == id, traffic_light_states, 0)
        )
        traffic_light_column = jnp.full_like(roadgraph_points_x, traffic_light_state, dtype=jnp.float32)

        polyline = jnp.stack(
            [
                roadgraph_points_x,
                roadgraph_points_y,
                roadgraph_heading,
                traffic_light_column,
                lane_type.astype(jnp.float32),
            ],
            axis=1,
        )

        sort_key = jnp.where(point_mask, 0, 1)
        order = jnp.argsort(sort_key, stable=True)
        polyline = jnp.take(polyline, order, axis=0)

        polyline_len = jnp.sum(point_mask.astype(jnp.int32))
        safe_polyline_len = jnp.maximum(polyline_len, 1)
        sampled_points = jnp.round(
            jnp.linspace(0, safe_polyline_len - 1, num_points_polyline)
        ).astype(jnp.int32)
        cur_polyline = jnp.take(polyline, sampled_points, axis=0)
        cur_valid = ((id != -1) & (polyline_len > 0)).astype(jnp.int32)
        cur_polyline = jnp.where(cur_valid > 0, cur_polyline, 0.0)

        polylines.append(cur_polyline)
        polylines_valid.append(cur_valid)

    polylines = jnp.stack(polylines, axis=0)
    polylines_valid = jnp.stack(polylines_valid, axis=0)

    return {
        "polylines": polylines,
        "polylines_valid": polylines_valid,
    }


@jax.jit(
    static_argnames=[
        "current_index",
        "use_full_agent_info",
        "max_polylines",
        "num_points_polyline",
        "use_log",
        "remove_history",
    ]
)
def data_process_scenarios(
    scenarios,
    current_index=10,
    use_full_agent_info=True,
    max_polylines=256,
    num_points_polyline=30,
    use_log=True,
    remove_history=False,
):
    traffic_info = data_process_traffic_light(scenarios, current_index=current_index)
    agents_info = data_process_agent(
        scenarios, current_index=current_index, use_full_agent_info=use_full_agent_info
    )
    # map_info = data_process_map(
    #     scenarios,
    #     traffic_info,
    #     agents_info,
    #     max_polylines=max_polylines,
    #     num_points_polyline=num_points_polyline,
    # )
    data_dict = {}
    data_dict.update(traffic_info)
    data_dict.update(agents_info)
    #data_dict.update(map_info)
    return data_dict
