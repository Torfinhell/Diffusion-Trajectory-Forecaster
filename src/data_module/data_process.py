import jax
import jax.numpy as jnp
import tensorflow as tf
from waymax import dataloader, datatypes

COORD_SCALE = 1.0


def wrap_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jax.jit(static_argnames=["x_index", "y_index", "heading_index"])
def batch_transform_trajs_to_local_frame(
    trajs,
    origin_xy,
    origin_theta,
    x_index=0,
    y_index=1,
    heading_index=5,
):
    if trajs.ndim == 3:
        trajs = trajs[None, ...]
        squeeze_batch = True
    elif trajs.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_trajs_to_local_frame expected 3D or 4D input, "
            f"got {trajs.shape}"
        )

    x = trajs[..., x_index] # (B, num_agents, num_timesteps)
    y = trajs[..., y_index]
    theta = trajs[..., heading_index]
    origin_xy = origin_xy[None, ...]      # (1, num_agents)
    origin_theta = origin_theta[None, ...] 
    origin_x = origin_xy[..., 0][..., None]# (1, num_agents, 1)
    origin_y = origin_xy[..., 1][..., None]
    origin_theta = origin_theta[..., None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)
    dx = x - origin_x
    dy = y - origin_y
    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta
    local_theta = wrap_angle(theta - origin_theta)

    transformed = trajs.at[..., x_index].set(local_x)
    transformed = transformed.at[..., y_index].set(local_y)
    transformed = transformed.at[..., heading_index].set(local_theta)

    valid_mask = jnp.any(trajs[..., :3] != 0, axis=-1, keepdims=True)
    transformed = jnp.where(valid_mask, transformed, 0.0)
    if squeeze_batch:
        return transformed[0]
    return transformed

def batch_transform_polylines_to_local_frame(polylines):
    if polylines.ndim == 3:
        polylines = polylines[None, ...]
        squeeze_batch = True
    elif polylines.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_polylines_to_local_frame expected 3D or 4D input, "
            f"got {polylines.shape}"
        )

    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]
    origin_x = x[:, :, 0, None]
    origin_y = y[:, :, 0, None]
    origin_theta = theta[:, :, 0, None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)
    dx = x - origin_x
    dy = y - origin_y
    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta
    local_theta = wrap_angle(theta - origin_theta)
    local_polylines = jnp.stack([local_x, local_y, local_theta], axis=-1)
    valid_mask = jnp.any(polylines[..., :3] != 0, axis=-1, keepdims=True)
    local_polylines = jnp.where(valid_mask, local_polylines, 0.0)
    transformed = jnp.concatenate([local_polylines, polylines[..., 3:]], axis=-1)
    if squeeze_batch:
        return transformed[0]
    return transformed

@jax.jit(static_argnames=["x_index", "y_index", "heading_index"])
def batch_transform_trajs_to_global_frame(
    trajs,
    origin_xy,
    origin_theta,
    x_index=0,
    y_index=1,
    heading_index=None,
):
    if trajs.ndim == 3:
        trajs = trajs[None, ...]
        origin_xy = origin_xy[None, ...]
        origin_theta = origin_theta[None, ...]
        squeeze_batch = True
    elif trajs.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "batch_transform_trajs_to_global_frame expected 3D or 4D input, "
            f"got {trajs.shape}"
        )

    local_x = trajs[..., x_index]
    local_y = trajs[..., y_index]
    origin_x = origin_xy[..., 0][..., None]
    origin_y = origin_xy[..., 1][..., None]
    origin_theta = origin_theta[..., None]
    cos_theta = jnp.cos(origin_theta)
    sin_theta = jnp.sin(origin_theta)

    global_x = local_x * cos_theta - local_y * sin_theta + origin_x
    global_y = local_x * sin_theta + local_y * cos_theta + origin_y

    transformed = trajs.at[..., x_index].set(global_x)
    transformed = transformed.at[..., y_index].set(global_y)

    if heading_index is not None and trajs.shape[-1] > heading_index:
        global_theta = wrap_angle(trajs[..., heading_index] + origin_theta)
        transformed = transformed.at[..., heading_index].set(global_theta)

    if squeeze_batch:
        return transformed[0]
    return transformed

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
        top_idx = jnp.argpartition(valid_dist, topk-1, axis=-1)[..., :topk]

        roadgraph_ids = jnp.broadcast_to(
        roadgraph.ids, top_idx.shape[:-1] + (roadgraph.ids.shape[-1],)
        )
        return jnp.take_along_axis(roadgraph_ids, top_idx, axis=-1)

    else:
        return roadgraph.ids

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
    origin_theta = jnp.take_along_axis(
        traj.yaw[..., : current_index + 1],
        safe_last_valid_idx[..., None],
        axis=-1,
    ).squeeze(axis=-1)
    origin_theta = jnp.where(has_history, origin_theta, 0.0)

    agents_info = batch_transform_trajs_to_local_frame(
    agents_info,
    origin_xy=origin_xyz[..., :2],
    origin_theta=origin_theta,
)

    valid_mask = traj.valid[..., None]
    agents_info = jnp.where(valid_mask & has_history[..., None, None], agents_info, 0.0)

    agent_past = agents_info[..., : current_index + 1, :]
    agent_future = agents_info[..., current_index + 1 :, :]
    # has_history is used here
    agent_future_valid = (
        traj.valid[..., current_index + 1 :, None] & has_history[..., None, None]
    )

    is_modeled = scenarios.object_metadata.is_modeled
    is_interesting = scenarios.object_metadata.objects_of_interest
    is_valid = scenarios.object_metadata.is_valid
    agents_coeffs = jnp.where(is_modeled & is_interesting, 10.0, 1.0)
    agents_coeffs = jnp.where(is_valid, agents_coeffs, 0.0)

    return {
        "agent_past": agent_past,
        "agent_future": agent_future,
        "agent_future_valid": agent_future_valid,
        "agents_valid": has_history & is_valid,
        "agents_coeffs": agents_coeffs,
        "agents_types": scenarios.object_metadata.object_types,
        "origin_xy": origin_xyz[..., :2],
        "origin_theta": origin_theta,
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
    agent_positions = batch_transform_trajs_to_global_frame(trajs=agent_past[..., :2], 
                                                            origin_xy=agents_info["origin_xy"], 
                                                            origin_theta=agents_info["origin_theta"])
    agent_positions = agent_positions[:, -1, :]
    # this filter is faster then the one in waymax 
    map_ids = filter_topk_roadgraph_points(roadgraph_points, agent_positions, 1000)
    map_ids = jnp.where(current_valid[:, None], map_ids, -1)

    ordered_map_ids = map_ids.T.reshape(-1)

    def collect_unique_ids(ordered_map_ids, max_polylines):
        n = ordered_map_ids.shape[0]
        indices = jnp.arange(n)
        is_duplicate = jnp.any(
            (ordered_map_ids[:, None] == ordered_map_ids[None, :])
            & (indices[:, None] > indices[None, :]),
            axis=1,
        )
        is_valid = (ordered_map_ids != -1) & ~is_duplicate
        sort_idx = jnp.argsort((~is_valid).astype(jnp.int32), stable=True)
        return ordered_map_ids[sort_idx][:max_polylines]
    
    sorted_map_ids = collect_unique_ids(ordered_map_ids, max_polylines)

    roadgraph_points_x = roadgraph_points.x
    roadgraph_points_y = roadgraph_points.y
    roadgraph_points_dir_x = roadgraph_points.dir_x
    roadgraph_points_dir_y = roadgraph_points.dir_y
    roadgraph_points_types = roadgraph_points.types
    roadgraph_heading = jnp.arctan2(roadgraph_points_dir_y, roadgraph_points_dir_x)
    base_features = jnp.stack(
        [
            roadgraph_points_x,
            roadgraph_points_y,
            roadgraph_heading,
        ],
        axis=1,
    )

    def build_polyline(id):
        point_mask = (roadgraph_points.ids == id) & roadgraph_points.valid
        lane_type = jnp.where(point_mask, roadgraph_points_types, 0)
        traffic_light_state = jnp.max(
            jnp.where(traffic_lane_ids == id, traffic_light_states, 0)
        )
        traffic_light_column = jnp.full_like(
            roadgraph_points_x, traffic_light_state, dtype=jnp.float32
        )
        polyline = jnp.concatenate(
            [
                base_features,
                traffic_light_column[:, None],
                lane_type.astype(jnp.float32)[:, None],
            ],
            axis=1,
        )
        # [num_points, 5]

        #moving valid points to the front
        sort_key = jnp.where(point_mask, 0, 1)
        order = jnp.argsort(sort_key, stable=True)
        polyline = jnp.take(polyline, order, axis=0)

        polyline_len = jnp.sum(point_mask.astype(jnp.int32))
        safe_polyline_len = jnp.maximum(polyline_len, 1)
        sampled_points = jnp.round(
            jnp.linspace(0, safe_polyline_len - 1, num_points_polyline)
        ).astype(jnp.int32)
        cur_polyline = jnp.take(polyline, sampled_points, axis=0)
        return cur_polyline, 1

    polylines, polylines_valid = jax.lax.map(build_polyline, sorted_map_ids)
    polylines = batch_transform_polylines_to_local_frame(polylines)
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
    map_info = data_process_map(
        scenarios,
        traffic_info,
        agents_info,
        max_polylines=max_polylines,
        num_points_polyline=num_points_polyline,
    )
    data_dict = {}
    data_dict.update(traffic_info)
    data_dict.update(agents_info)
    data_dict.update(map_info)
    return data_dict


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
def data_process_scenarios_batch(
    scenarios,
    current_index=10,
    use_full_agent_info=True,
    max_polylines=256,
    num_points_polyline=30,
    use_log=True,
    remove_history=False,
):
    return jax.vmap(
        lambda scenario: data_process_scenarios(
            scenario,
            current_index=current_index,
            use_full_agent_info=use_full_agent_info,
            max_polylines=max_polylines,
            num_points_polyline=num_points_polyline,
            use_log=use_log,
            remove_history=remove_history,
        )
    )(scenarios)
