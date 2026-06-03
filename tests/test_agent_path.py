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
    obj = AgentPath(path, action_len=5, ref_idx=-1)
    local_xy = obj.to_local()[..., :2]
    restored_xy = obj.xy_to_global(local_xy)
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
    obj = AgentPath(full_path, action_len=action_len, ref_idx=0)
    recovered_actions, _ = obj.actions()
    assert jnp.allclose(recovered_actions, actions, atol=5e-3)


def test_rollout_reconstructs_local_future_xy():
    action_len = 5
    actions = jnp.array([[[0.1, 0.03], [0.0, -0.01]]], dtype=jnp.float32)
    current_state = jnp.array([[0.0, 0.0, 0.0, 3.0, 0.0]], dtype=jnp.float32)
    rollout = roll_out(current_state, actions, action_len=action_len, global_frame=True)
    full_path = jnp.concatenate([current_state[:, None, :], rollout], axis=1)

    obj = AgentPath(full_path, action_len=action_len, ref_idx=0)
    recovered_actions, _ = obj.actions()
    pred_local_xy = obj.rollout_actions(
        recovered_actions, accel_scale=1.0, yaw_rate_scale=1.0
    )
    gt_local_xy = obj.to_local()[..., 1:, :2]
    assert pred_local_xy.shape == gt_local_xy.shape
    assert jnp.allclose(pred_local_xy, gt_local_xy, atol=1e-3)
