import jax.numpy as jnp

from src.data_module import AgentPath
from src.utils import roll_out


def test_local_global_roundtrip_xy():
    path = jnp.array(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0, 1.0, 0.0],
                [4.0, 0.0, 0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    # anchor at last timestep by providing explicit ref_coords
    ref_coords = path[:, -1, :]
    obj = AgentPath(path, action_len=5, ref_coords=ref_coords)
    local_xy = obj.to_local()[..., :2]
    # convert back to global using ref_coords
    anchor = obj.ref_coords
    x0 = anchor[..., 0][:, None]
    y0 = anchor[..., 1][:, None]
    theta0 = anchor[..., 2][:, None]
    cos_t = jnp.cos(theta0)
    sin_t = jnp.sin(theta0)
    g_x = local_xy[..., 0] * cos_t + -local_xy[..., 1] * sin_t + x0
    g_y = local_xy[..., 0] * sin_t + local_xy[..., 1] * cos_t + y0
    restored_xy = jnp.stack([g_x, g_y], axis=-1)
    assert jnp.allclose(restored_xy, path[..., :2], atol=1e-5)


def test_inverse_kinematics_recovers_piecewise_actions():
    action_len = 5
    actions = jnp.array(
        [
            [[0.2, 0.05], [0.0, -0.02]],
            [[0.1, 0.00], [-0.1, 0.01]],
        ],
        dtype=jnp.float32,
    )  # (A=2, K=2, 2)
    current_state = jnp.array(
        [
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.2, 1.5, 0.0],
        ],
        dtype=jnp.float32,
    )
    rollout = roll_out(
        current_state=current_state,
        actions=actions,
        action_len=action_len,
        global_frame=True,
    )
    full_path = jnp.concatenate([current_state[:, None, :], rollout], axis=1)
    obj = AgentPath(full_path, action_len=action_len, ref_coords=full_path[:, 0, :])
    recovered_actions, _ = obj.actions()
    assert jnp.allclose(recovered_actions, actions, atol=5e-3)


def test_rollout_reconstructs_local_future_xy():
    action_len = 5
    actions = jnp.array([[[0.1, 0.03], [0.0, -0.01]]], dtype=jnp.float32)
    current_state = jnp.array([[0.0, 0.0, 0.0, 3.0, 0.0]], dtype=jnp.float32)
    rollout = roll_out(current_state, actions, action_len=action_len, global_frame=True)
    full_path = jnp.concatenate([current_state[:, None, :], rollout], axis=1)

    obj = AgentPath(full_path, action_len=action_len, ref_coords=full_path[:, 0, :])
    recovered_actions, _ = obj.actions()
    from src.utils.path_kinematics import roll_out

    rolled = roll_out(
        obj.current_state_for_rollout(),
        recovered_actions,
        action_len=action_len,
        dt=obj.dt,
        global_frame=False,
    )
    pred_local_xy = rolled[..., :2]
    gt_local_xy = obj.to_local()[..., 1:, :2]
    assert pred_local_xy.shape == gt_local_xy.shape
    assert jnp.allclose(pred_local_xy, gt_local_xy, atol=1e-3)
