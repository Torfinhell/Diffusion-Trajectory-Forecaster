import jax
import jax.numpy as jnp
import tensorflow as tf
from waymax import dataloader, datatypes

COORD_SCALE = 1.0


@jax.jit(static_argnames=["current_index"])
def data_process_traffic_light(
    scenarios,
    current_index=10,
):
    """
    Process traffic light data from the given scenarios.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing traffic light information.

    Returns:
        tuple: A tuple containing the processed traffic light points, lane IDs, and states.
    """
    traffic_lights = scenarios.log_traffic_light

    ############# Get Traffic Lights #############
    traffic_lane_ids = traffic_lights.lane_ids[:, current_index]
    traffic_light_states = traffic_lights.state[:, current_index]
    traffic_stop_points = traffic_lights.xy[:, current_index]
    traffic_light_valid = traffic_lights.valid[:, current_index]

    traffic_light_points = jnp.concatenate(
        [traffic_stop_points, traffic_light_states[..., None]], axis=2
    )
    traffic_light_points = jnp.float32(traffic_light_points)
    traffic_light_points = jnp.where(
        traffic_light_valid[..., None], traffic_light_points, 0.0
    )
    return traffic_light_points, traffic_lane_ids, traffic_light_states


@jax.jit(
    static_argnames=[
        "max_num_objects",
        "max_polylines",
        "current_index",
        "num_points_polyline",
        "use_log",
        "remove_history",
    ]
)
def data_process_scenarios(
    scenarios,
    max_num_objects=64,
    max_polylines=256,
    current_index=10,
    num_points_polyline=30,
    use_log=True,
    remove_history=False,
    model_type="linear",
):
    data_dict = {}
    (traffic_light_points, traffic_lane_ids, traffic_light_states) = (
        data_process_traffic_light(
            scenarios,
            current_index=current_index,
        )
    )
    traj = scenarios.log_trajectory
    context_xy = jnp.stack(
        [traj.x[..., :current_index], traj.y[..., :current_index]],
        axis=-1,
    )
    context_valid = traj.valid[..., :current_index, None]
    gt_xy = jnp.stack(
        [traj.x[..., current_index:], traj.y[..., current_index:]],
        axis=-1,
    )
    gt_valid = traj.valid[..., current_index:, None]

    history_valid = traj.valid[..., :current_index]
    history_idx = jnp.arange(current_index, dtype=jnp.int32)
    last_valid_idx = jnp.max(jnp.where(history_valid, history_idx, -1), axis=-1)
    has_history = last_valid_idx >= 0
    safe_last_valid_idx = jnp.maximum(last_valid_idx, 0)
    origin_xy = jnp.take_along_axis(
        traj.xy[..., :current_index, :],
        safe_last_valid_idx[..., None, None],
        axis=-2,
    )
    origin_xy = jnp.where(has_history[..., None, None], origin_xy, 0.0)
    # past trajectory (context)
    data_dict.update(
        {
            "context": jnp.where(context_valid, context_xy - origin_xy, 0.0)
            / COORD_SCALE
        }
    )
    # future trajectory (target for diffusion)
    data_dict.update(
        {
            "gt_xy": jnp.where(
                gt_valid & has_history[..., None, None], gt_xy - origin_xy, 0.0
            )
            / COORD_SCALE
        }
    )

    # mask for valid objects
    data_dict.update(
        {
            "gt_xy_mask": jnp.repeat(
                (traj.valid[..., current_index:, None] & has_history[..., None, None]),
                2,
                axis=-1,
            )  # [N,H,2]
        }
    )
    data_dict.update(
        {
            "agents_type": scenarios.object_metadata.object_types,
            "origin_xy": origin_xy,
            "coord_scale": jnp.full(origin_xy.shape[:-1] + (1,), COORD_SCALE),
        }
    )
    return data_dict
