"""Velocity-lag point-mass drone dynamics (simplified, no attitude/motors)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DynamicsConfig:
    """Physical parameters of the simplified drone. Units: m, s, rad."""

    dt: float = 0.05
    tau_vel: float = 0.3
    tau_yaw: float = 0.15
    max_speed_xy: float = 3.0
    max_speed_z: float = 2.0
    max_yaw_rate: float = 1.5
    max_accel: float = 4.0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DynamicsConfig":
        """Build a config from a dict (e.g. the `dynamics` section of a YAML file)."""
        return cls(**params)


@dataclass
class DroneState:
    """Full simulated state. Position and velocity are in the world frame."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class VelocityLagDynamics:
    """Drone that tracks commanded velocities with first-order lag.

    Action (all in [-1, 1], values outside are clipped):
        [forward, lateral, vertical, yaw_rate], forward/lateral in the body frame.
    """

    def __init__(self, config: DynamicsConfig) -> None:
        self.config = config
        self.state = DroneState()
        self._alpha_vel = 1.0 - math.exp(-config.dt / config.tau_vel)
        self._alpha_yaw = 1.0 - math.exp(-config.dt / config.tau_yaw)

    def reset(self, position: np.ndarray, yaw: float = 0.0) -> DroneState:
        """Place the drone at rest at `position` facing `yaw`."""
        self.state = DroneState(
            position=np.asarray(position, dtype=np.float64).copy(),
            velocity=np.zeros(3),
            yaw=wrap_angle(float(yaw)),
            yaw_rate=0.0,
        )
        return self.state

    def step(self, action: np.ndarray) -> DroneState:
        """Advance the simulation by one time step and return the new state."""
        cfg = self.config
        s = self.state
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Scale the normalized action to physical commands (body frame).
        cmd_body = np.array(
            [a[0] * cfg.max_speed_xy, a[1] * cfg.max_speed_xy, a[2] * cfg.max_speed_z]
        )
        cmd_yaw_rate = a[3] * cfg.max_yaw_rate

        # Rotate the horizontal command from the body frame to the world frame.
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        cmd_world = np.array(
            [c * cmd_body[0] - sn * cmd_body[1], sn * cmd_body[0] + c * cmd_body[1], cmd_body[2]]
        )

        # First-order lag toward the commanded velocity, with an acceleration limit.
        dv = self._alpha_vel * (cmd_world - s.velocity)
        accel = dv / cfg.dt
        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > cfg.max_accel:
            dv = dv * (cfg.max_accel / accel_norm)
        s.velocity = s.velocity + dv

        # Yaw rate lags the same way; yaw integrates the yaw rate.
        s.yaw_rate += self._alpha_yaw * (cmd_yaw_rate - s.yaw_rate)
        s.yaw = wrap_angle(s.yaw + s.yaw_rate * cfg.dt)

        # Position integrates the new velocity (semi-implicit Euler).
        s.position = s.position + s.velocity * cfg.dt
        return s