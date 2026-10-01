"""Plot an episode's trajectory (top-down and side views)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from uav_nav.envs.nav_env import UavNavEnv


def _panel(ax, env: UavNavEnv, traj: np.ndarray, i: int, j: int, xlabel: str, ylabel: str) -> None:
    ax.add_patch(Rectangle((0, 0), env.arena[i], env.arena[j], fill=False, edgecolor="black"))
    for ox, oy, oz, r in env.obstacles:
        c = (ox, oy, oz)
        ax.add_patch(Circle((c[i], c[j]), r, color="tab:red", alpha=0.35))  # sphere projects to a circle
    ax.add_patch(Circle((env.goal[i], env.goal[j]), env.task["goal_tolerance"], color="tab:green", alpha=0.5))
    ax.plot(traj[:, i], traj[:, j], color="tab:blue", linewidth=1.5)
    ax.plot(traj[0, i], traj[0, j], "ko", markersize=5, label="start")
    ax.set_xlim(-0.5, env.arena[i] + 0.5)
    ax.set_ylim(-0.5, env.arena[j] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_episode(env: UavNavEnv, path: str | Path, title: str | None = None) -> Path:
    """Save a PNG of the env's current episode trajectory and return its path."""
    traj = np.asarray(env.trajectory)
    fig = Figure(figsize=(11, 5))
    ax_top, ax_side = fig.subplots(1, 2)
    _panel(ax_top, env, traj, 0, 1, "x (m)", "y (m)")
    _panel(ax_side, env, traj, 0, 2, "x (m)", "z (m)")
    ax_top.set_title("Top view (start = black dot, goal = green)")
    ax_side.set_title("Side view")
    if title:
        fig.suptitle(title)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out