import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["hatch.linewidth"] = 4  # previous svg hatch linewidth
from typing import Dict

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, RegularPolygon

from .vis_config_bright import (
    canvas_config,
    crosswalk_config,
    road_edge_config,
    road_line_config,
    speed_bump_config,
    stop_sign_config,
)

v_max = 10
v_min = 0


def setup_canvas():
    fig = plt.figure(figsize=(canvas_config["width"], canvas_config["width"]))
    ax = fig.add_subplot(111)
    ax.set_facecolor(canvas_config["background_color"])
    ax.set_aspect("equal")
    if not canvas_config["tick_on"]:
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labeltop=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()  # pad=0)
    return fig, ax


def _plot_line(
    points: np.ndarray,
    config: Dict,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override config
    color = config["color"] if color is None else color
    linewidth = config["linewidth"] if linewidth is None else linewidth
    linestyle = config["linestyle"] if linestyle is None else linestyle
    alpha = config["alpha"] if alpha is None else alpha

    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=2,
    )


def _plot_broken_line(
    points: np.ndarray,
    config: Dict,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override config
    color = config["color"] if color is None else color
    linewidth = config["linewidth"] if linewidth is None else linewidth
    linestyle = config["linestyle"] if linestyle is None else linestyle
    alpha = config["alpha"] if alpha is None else alpha

    n_broken = 8
    skip = 2
    n_points = int(points.shape[0] / n_broken) * n_broken
    point_x = points[:n_points, 0].reshape(-1, n_broken).T
    point_y = points[:n_points, 1].reshape(-1, n_broken).T

    ax.plot(
        point_x[:-skip, :],
        point_y[:-skip, :],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )


def plot_road_line(
    points: np.ndarray,
    line_type: str,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    config = road_line_config[line_type]

    if "BROKEN" in line_type:
        _plot_broken_line(
            points=points,
            config=config,
            ax=ax,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
    else:
        _plot_line(
            points=points,
            config=config,
            ax=ax,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )


def plot_road_edge(
    points: np.ndarray,
    line_type: str,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    config = road_edge_config[line_type]

    _plot_line(
        points=points,
        config=config,
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )


def plot_speed_bump(
    points: np.ndarray,
    ax: plt.Axes = None,
    facecolor: str = None,
    edgecolor: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = speed_bump_config["facecolor"] if facecolor is None else facecolor
    edgecolor = speed_bump_config["edgecolor"] if edgecolor is None else edgecolor
    alpha = speed_bump_config["alpha"] if alpha is None else alpha

    p = Polygon(
        points,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0,
        alpha=alpha,
        hatch=r"//",
        zorder=2,
    )

    ax.add_patch(p)


def plot_crosswalk(
    points,
    ax: plt.Axes = None,
    facecolor: str = None,
    edgecolor: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = crosswalk_config["facecolor"] if facecolor is None else facecolor
    edgecolor = crosswalk_config["edgecolor"] if edgecolor is None else edgecolor
    alpha = crosswalk_config["alpha"] if alpha is None else alpha

    p = Polygon(
        points,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
        alpha=alpha,
        hatch=r"//",
        zorder=2,
    )

    ax.add_patch(p)


def plot_stop_sign(
    point: np.ndarray,
    ax: plt.Axes = None,
    radius: float = None,
    facecolor: str = None,
    edgecolor: str = None,
    linewidth: float = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = stop_sign_config["facecolor"] if facecolor is None else facecolor
    edgecolor = stop_sign_config["edgecolor"] if edgecolor is None else edgecolor
    linewidth = stop_sign_config["linewidth"] if linewidth is None else linewidth
    radius = stop_sign_config["radius"] if radius is None else radius
    alpha = stop_sign_config["alpha"] if alpha is None else alpha

    point = point.reshape(-1)

    p = RegularPolygon(
        point,
        numVertices=6,
        radius=radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )

    ax.add_patch(p)


def plot_traj_with_speed(
    trajs: np.ndarray,
    speeds: np.ndarray,
    valids: np.ndarray,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fixed_linewidth: float = None,
    fixed_linestyle: str = None,
    fixed_alpha: float = None,
    show_colorbar: bool = False,
    v_min: float = 0,
    v_max: float = 10,
):
    # print(v_min, v_max)
    """
    This function plot trajectory with speed as color gradient
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # plot color line
    norm = plt.Normalize(v_min, v_max)
    A, T, _ = trajs.shape
    # traj have feature [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
    for a in range(A):
        points = trajs[a]
        speed = speeds[a]
        valid = valids[a]
        points = points[valid, :]
        segments = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
        # override config
        linewidth = 3 if fixed_linewidth is None else fixed_linewidth
        linestyle = "-" if fixed_linestyle is None else fixed_linestyle
        alpha = 0.8 if fixed_alpha is None else fixed_alpha

        lc = LineCollection(
            segments,
            cmap="inferno",
            norm=norm,
            linestyle=linestyle,
            alpha=alpha,
            zorder=3,
        )
        # Set the values used for colormapping
        lc.set_array(speed)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
    if show_colorbar:
        fig.colorbar(
            line, ax=ax, label="speed (m/s)", location="bottom", shrink=0.3, pad=0.02
        )
