from src import visualization
import imageio
import io
import tempfile
import mediapy
import numpy as np
from contextlib import nullcontext
from matplotlib import pyplot as plt, cm as mpl_cm
from PIL import Image as PILImage, ImageDraw

#stay unused for now

def plot_state(
    current_state,
    log_traj = False,
    traj_preds=None,
    traj_pred_score=None,
    past_traj_length = 0,
    dx = 75,
    center_agent_idx = -1,
    filename = None,
    t = None,
    tick_off = False,
    return_ax = False,
    img_size = (400,400),
    font_size = 12,
    center_xy = None,
    traj_color = 'r',
    is_ego = None,
    is_adv = None,
    time_idx = 10,
):
    viz_config = visualization.utils.VizConfig()
    fig, ax = visualization.utils.init_fig_ax(viz_config)
    if log_traj:
        traj = current_state.log_trajectory
    else:
        traj = current_state.sim_trajectory
    indices = np.arange(traj.num_objects)
    is_controlled = current_state.object_metadata.is_controlled

    visualization.plot_trajectory(
        ax, traj, is_controlled, time_idx=time_idx,
        indices=indices, past_traj_length = past_traj_length,
        is_ego = is_ego, is_adv = is_adv,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    visualization.plot_roadgraph_points(ax, current_state.roadgraph_points, verbose=False)
    visualization.plot_traffic_light_signals_as_points(
        ax, current_state.log_traffic_light, time_idx, verbose=False
    )

    current_xy = traj.xy[:, time_idx, :]
    if center_xy is not None:
        origin_x, origin_y = center_xy
    elif center_agent_idx == -1:
        xy = current_xy[current_state.object_metadata.is_sdc]
        origin_x, origin_y = xy[0, :2]
    else:
        xy = current_xy[center_agent_idx]
        origin_x, origin_y = xy[:2]
    # Zoom
    
    ax.axis((
        origin_x - dx,
        origin_x + dx,
        origin_y - dx,
        origin_y + dx,
    ))
    if t is None:
        t = (time_idx - 10) / 10
    if font_size>0:
        ax.text(origin_x - 0.9*dx, origin_y + 0.9*dx, f"t={t:.1f} s", fontsize=font_size)
    
    if tick_off:
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)

    if traj_preds is not None:
        preds = np.asarray(traj_preds)
        A, T, D = preds.shape
        colors = plt.cm.tab20(np.linspace(0, 1, max(A, 1)))

        if traj_pred_score is not None:
            scores = np.asarray(traj_pred_score).reshape(A)
            for i, (traj, score) in enumerate(zip(preds, scores)):
                if score < 0.01:
                    continue
                ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=2.5, alpha=score * 0.8 + 0.2, zorder=10)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=30, zorder=11)
        else:
            for i, traj in enumerate(preds):
                ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=2.5, alpha=0.85, zorder=10)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=30, zorder=11)
        
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
    )
    if filename is not None:
        fig.savefig(filename,
                    bbox_inches='tight',
                    transparent=False,
                    pad_inches=0.02)
    if return_ax:
        return fig, ax
    return mediapy.resize_image(visualization.utils.img_from_fig(fig), img_size)


def create_prediction_gif(
    sample: dict,
    pred_xy_world: np.ndarray,
    fps: int = 10,
    mpl_lock=None,
) -> str:
    """
    Animate the scene: past frames via Waymax renderer, future frames via fast
    numpy compositing — agent boxes drawn as filled polygons on a static background.

    pred_xy_world: (A, T_future, 2) world-frame predicted positions
    mpl_lock: optional lock to serialize matplotlib calls across threads
    Returns: path to a temporary GIF file.
    """
    lock = mpl_lock or nullcontext()

    scenario = sample.get("scenario")
    A, T_future, _ = pred_xy_world.shape
    GIF_SIZE = (300, 300)
    W, H = GIF_SIZE
    frames = []

    # Past: subsampled Waymax renders (every 2nd timestep)
    for t in range(0, 11, 2):
        with lock:
            frames.append(plot_state(
                current_state=scenario, log_traj=True, dx=75,
                tick_off=True, img_size=GIF_SIZE, time_idx=t,
            ))

    # Render static background at t=10 once, capture axis limits for world→pixel.
    with lock:
        fig, ax = plot_state(
            current_state=scenario, log_traj=True, dx=75,
            tick_off=True, img_size=GIF_SIZE, time_idx=10, return_ax=True,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=80)
        plt.close(fig)
    buf.seek(0)
    bg_pil = PILImage.open(buf).convert("RGB").resize(GIF_SIZE)
    bg_arr = np.array(bg_pil)
    frames.append(bg_arr)  # t=10 frame

    def world_to_pixel(xy):
        px = (xy[..., 0] - xlim[0]) / (xlim[1] - xlim[0]) * W
        py = (1.0 - (xy[..., 1] - ylim[0]) / (ylim[1] - ylim[0])) * H
        return np.stack([px, py], axis=-1)

    # Precompute per-agent box half-dims from t=10 (size doesn't change).
    log_traj = scenario.log_trajectory
    half_l = np.array(log_traj.length[:, 10]) / 2   # (A,)
    half_w = np.array(log_traj.width[:, 10]) / 2    # (A,)
    valid = np.array(log_traj.valid[:, 10])          # (A,)
    fallback_yaw = np.array(log_traj.yaw[:, 10])    # (A,) used for t=0 step

    # Precompute per-step yaw from consecutive predicted positions (atan2 of displacement).
    # Shape: (A, T_future). For t=0 fall back to the log yaw at t=10.
    pred_yaw = np.empty((A, T_future), dtype=np.float32)
    pred_yaw[:, 0] = fallback_yaw
    if T_future > 1:
        dx = pred_xy_world[:, 1:, 0] - pred_xy_world[:, :-1, 0]  # (A, T-1)
        dy = pred_xy_world[:, 1:, 1] - pred_xy_world[:, :-1, 1]
        moving = np.hypot(dx, dy) > 0.01
        computed = np.arctan2(dy, dx)
        pred_yaw[:, 1:] = np.where(moving, computed, pred_yaw[:, :T_future-1])
        for t in range(1, T_future):
            pred_yaw[:, t] = np.where(moving[:, t-1], computed[:, t-1], pred_yaw[:, t-1])

    colors = mpl_cm.tab20(np.linspace(0, 1, max(A, 1)))
    color_rgb = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]

    corners_local = np.array([[ 1,  1], [ 1, -1], [-1, -1], [-1,  1]], dtype=np.float32)

    for t in range(0, T_future, 2):
        frame = PILImage.fromarray(bg_arr.copy())
        draw = ImageDraw.Draw(frame)

        for i in range(A):
            if not valid[i]:
                continue
            cx, cy = pred_xy_world[i, t]
            yaw = pred_yaw[i, t]
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            scaled = corners_local * np.array([half_l[i], half_w[i]])
            rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            world_corners = scaled @ rot.T + np.array([cx, cy])
            px_corners = world_to_pixel(world_corners)
            poly = [(float(p[0]), float(p[1])) for p in px_corners]
            draw.polygon(poly, fill=color_rgb[i], outline=(0, 0, 0))

        frames.append(np.array(frame))

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    imageio.mimsave(tmp.name, frames, fps=fps, loop=0)
    return tmp.name