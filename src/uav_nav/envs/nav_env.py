"""Gymnasium environment: 3D UAV navigation with spherical obstacles."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_nav.dynamics.velocity_tracking import DynamicsConfig, VelocityLagDynamics
from uav_nav.envs.config import load_env_config


class UavNavEnv(gym.Env):
    """Fly a velocity-controlled quadrotor to a goal while avoiding spherical obstacles.

    Observation (34 dims with K=4, all clipped to [-1, 1]):
        goal position relative to drone, body frame (3)
        drone velocity, body frame (3)
        yaw as (sin, cos) (2)
        distance to walls: [x, y, z, size_x - x, size_y - y, size_z - z] / arena size (6)
        K nearest obstacles: relative position (3) + radius (1), zero-padded (4K)
        previous action (4)

    Action (4 dims in [-1, 1]): forward, lateral, vertical velocity, yaw rate.

    Episode ends (terminated) on goal, collision or leaving the arena;
    it is truncated after `task.max_steps` steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = config if config is not None else load_env_config()
        self.dynamics = VelocityLagDynamics(DynamicsConfig.from_dict(self.cfg["dynamics"]))
        self.arena = np.asarray(self.cfg["arena"]["size"], dtype=np.float64)
        self.drone_radius = float(self.cfg["drone"]["radius"])
        self.task = self.cfg["task"]
        self.obst_cfg = self.cfg["obstacles"]
        self.rw = self.cfg["reward"]
        self.k = int(self.cfg["observation"]["num_nearest_obstacles"])

        dyn = self.dynamics.config
        self._diag = float(np.linalg.norm(self.arena))
        self._vel_scale = np.array([dyn.max_speed_xy, dyn.max_speed_xy, dyn.max_speed_z])
        self._r_scale = max(float(self.obst_cfg["radius_max"]), 1e-6)

        obs_dim = 3 + 3 + 2 + 6 + 4 * self.k + 4
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.obstacles = np.zeros((0, 4))  # rows: x, y, z, radius
        self.goal = np.zeros(3)
        self.steps = 0
        self._prev_action = np.zeros(4)
        self._prev_dist = 0.0
        self.trajectory: list[np.ndarray] = []

    # ------------------------------------------------------------------ reset
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        `options` may contain "start" (3,), "goal" (3,), "yaw" (float) and
        "obstacles" (N, 4) to build fixed scenarios (used by baselines/evaluation).
        """
        super().reset(seed=seed)
        options = options or {}

        start = self._sample_point() if "start" not in options else np.asarray(options["start"], float)
        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float64)
        else:
            self.goal = self._sample_goal(start)
        yaw = float(options["yaw"]) if "yaw" in options else float(self.np_random.uniform(-math.pi, math.pi))

        if "obstacles" in options:
            self.obstacles = np.asarray(options["obstacles"], dtype=np.float64).reshape(-1, 4)
        else:
            self.obstacles = self._sample_obstacles(start, self.goal)

        self.dynamics.reset(start, yaw)
        self.steps = 0
        self._prev_action = np.zeros(4)
        self._prev_dist = float(np.linalg.norm(self.goal - start))
        self.trajectory = [start.copy()]
        clearance = self._min_clearance(start)
        return self._get_obs(), self._info(self._prev_dist, clearance, None)

    def _sample_point(self) -> np.ndarray:
        margin = float(self.task["spawn_margin"])
        return self.np_random.uniform(margin, self.arena - margin)

    def _sample_goal(self, start: np.ndarray) -> np.ndarray:
        min_dist = float(self.task["start_goal_min_distance"])
        for _ in range(1000):
            goal = self._sample_point()
            if np.linalg.norm(goal - start) >= min_dist:
                return goal
        raise RuntimeError("Could not sample a goal; check arena size / start_goal_min_distance.")

    def _sample_obstacles(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        n = int(self.np_random.integers(self.obst_cfg["num_min"], self.obst_cfg["num_max"] + 1))
        clearance = float(self.task["spawn_clearance"])
        placed: list[list[float]] = []
        for _ in range(n):
            for _attempt in range(50):
                r = float(self.np_random.uniform(self.obst_cfg["radius_min"], self.obst_cfg["radius_max"]))
                center = self.np_random.uniform(r, self.arena - r)
                if all(np.linalg.norm(center - p) - r >= clearance for p in (start, goal)):
                    placed.append([*center, r])
                    break
        return np.array(placed, dtype=np.float64).reshape(-1, 4)

    # ------------------------------------------------------------------- step
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one time step."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        state = self.dynamics.step(action)
        self.steps += 1
        self.trajectory.append(state.position.copy())

        dist = float(np.linalg.norm(self.goal - state.position))
        clearance = self._min_clearance(state.position)
        collision = clearance <= 0.0
        out_of_bounds = bool(np.any(state.position < 0.0) or np.any(state.position > self.arena))
        success = dist <= float(self.task["goal_tolerance"])

        # Priority: collision > out of bounds > success (conservative).
        termination: str | None = None
        if collision:
            termination = "collision"
        elif out_of_bounds:
            termination = "out_of_bounds"
        elif success:
            termination = "goal"
        terminated = termination is not None
        truncated = (not terminated) and self.steps >= int(self.task["max_steps"])
        if truncated:
            termination = "timeout"

        reward = self._reward(action, dist, clearance, termination)
        self._prev_dist = dist
        self._prev_action = action
        return self._get_obs(), float(reward), terminated, truncated, self._info(dist, clearance, termination)

    def _reward(
        self, action: np.ndarray, dist: float, clearance: float, termination: str | None
    ) -> float:
        rw = self.rw
        r = rw["w_progress"] * (self._prev_dist - dist) - rw["w_time"]
        if rw["w_safe"] > 0.0 and clearance < rw["safe_distance"]:
            r -= rw["w_safe"] * ((rw["safe_distance"] - clearance) / rw["safe_distance"]) ** 2
        r -= rw["w_control"] * float(np.sum(action**2))
        r -= rw["w_smooth"] * float(np.sum((action - self._prev_action) ** 2))
        if termination == "goal":
            r += rw["w_goal"]
        elif termination == "collision":
            r -= rw["w_collision"]
        elif termination == "out_of_bounds":
            r -= rw["w_bounds"]
        return r

    # ---------------------------------------------------------------- helpers
    def _to_body(self, v_world: np.ndarray) -> np.ndarray:
        """Rotate a world-frame vector into the drone's body frame (yaw only)."""
        c, s = math.cos(self.dynamics.state.yaw), math.sin(self.dynamics.state.yaw)
        return np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1], v_world[2]])

    def _clearances(self, position: np.ndarray) -> np.ndarray:
        """Distance from the drone's surface to each obstacle's surface (<= 0 means contact)."""
        if len(self.obstacles) == 0:
            return np.zeros(0)
        d = np.linalg.norm(self.obstacles[:, :3] - position, axis=1)
        return d - self.obstacles[:, 3] - self.drone_radius

    def _min_clearance(self, position: np.ndarray) -> float:
        c = self._clearances(position)
        return float(c.min()) if c.size else self._diag

    def _get_obs(self) -> np.ndarray:
        s = self.dynamics.state
        pos = s.position
        goal_rel = self._to_body(self.goal - pos) / self._diag
        vel = self._to_body(s.velocity) / self._vel_scale
        yaw_feat = np.array([math.sin(s.yaw), math.cos(s.yaw)])
        walls = np.clip(np.concatenate([pos / self.arena, (self.arena - pos) / self.arena]), 0.0, 1.0)

        obst = np.zeros((self.k, 4))
        if len(self.obstacles):
            for row, i in enumerate(np.argsort(self._clearances(pos))[: self.k]):
                obst[row, :3] = self._to_body(self.obstacles[i, :3] - pos) / self._diag
                obst[row, 3] = self.obstacles[i, 3] / self._r_scale

        obs = np.concatenate([goal_rel, vel, yaw_feat, walls, obst.ravel(), self._prev_action])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _info(self, dist: float, clearance: float, termination: str | None) -> dict[str, Any]:
        return {
            "distance_to_goal": dist,
            "min_obstacle_clearance": clearance,
            "termination": termination,
            "steps": self.steps,
        }