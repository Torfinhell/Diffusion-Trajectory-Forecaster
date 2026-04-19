import jax
import jax.numpy as jnp
import tensorflow as tf
from waymax import dataloader, datatypes

COORD_SCALE = 1.0


@jax.jit(static_argnames=["current_index"])
def data_process_traffic_light(scenarios, current_index=10):
    traffic_lights = scenarios.log_traffic_light
    traffic_lane_ids = traffic_lights.lane_ids[:, current_index]
    traffic_light_states = traffic_lights.state[:, current_index]
    traffic_stop_points = traffic_lights.xy[:, current_index]
    traffic_light_valid = traffic_lights.valid[:, current_index]

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

    current_xy = agents_info[..., current_index, :3]
    agents_info = agents_info.at[..., :3].set(
        agents_info[..., :3] - current_xy[..., None, :]
    )

    valid_mask = traj.valid[..., None]
    agents_info = jnp.where(valid_mask, agents_info, 0.0)

    agent_past = agents_info[..., : current_index + 1, :]
    agent_future = agents_info[..., current_index + 1 :, :]

    is_modeled = scenarios.object_metadata.is_modeled
    is_interesting = scenarios.object_metadata.objects_of_interest
    is_valid = scenarios.object_metadata.is_valid
    agents_coeffs = jnp.where(is_modeled & is_interesting, 10.0, 1.0)
    agents_coeffs = jnp.where(is_valid, agents_coeffs, 0.0)

    return {
        "agent_past": agent_past,
        "agent_future": agent_future,
        "agents_coeffs": agents_coeffs,
        "agents_types": scenarios.object_metadata.object_types,
    }


@jax.jit
def data_process_map(scenarios, traffic_info):
    return {}


@jax.jit(static_argnames=["current_index", "use_full_agent_info"])
def data_process_scenarios(scenarios, current_index=10, use_full_agent_info=True):
    traffic_info = data_process_traffic_light(scenarios, current_index=current_index)
    agents_info = data_process_agent(
        scenarios, current_index=current_index, use_full_agent_info=use_full_agent_info
    )
    map_info = data_process_map(scenarios, traffic_info)
    data_dict = {}
    data_dict.update(traffic_info)
    data_dict.update(agents_info)
    data_dict.update(map_info)
    return data_dict
