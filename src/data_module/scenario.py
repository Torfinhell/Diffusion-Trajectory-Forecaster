import jax.numpy as jnp

class scenario:
    def __init__(self, state, t=1, history=1, horizon=None):
        self.t = t
        self.history = history
        traj = state.log_trajectory        
        meta = state.object_metadata       
        if horizon is None:
            horizon = traj.xy.shape[1] - t
        self.horizon = horizon
        self.traj = traj
        
        obj_mask = meta.is_modeled & meta.is_valid

        # slices in time
        past_slice   = slice(t - history + 1, t + 1)          # length=history
        future_slice = slice(t + 1, t + 1 + horizon)          # length=horizon

        past_xy   = traj.xy[:, past_slice, :]      # (N, Hhist, 2) if you index objects first
        future_xy = traj.xy[:, future_slice, :]    # (N, Hfut, 2)
        past_valid   = traj.valid[:, past_slice]
        future_valid = traj.valid[:, future_slice]
        
        # apply object mask
        self.past_xy       = past_xy[obj_mask]
        self.future_xy     = future_xy[obj_mask]
        self.past_valid    = past_valid[obj_mask]
        self.future_valid  = future_valid[obj_mask]