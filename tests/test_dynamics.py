"""Unit tests for the velocity-lag dynamics."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_nav.dynamics.velocity_tracking import DynamicsConfig, VelocityLagDynamics

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "env" / "base.yaml"


def make(**overrides) -> VelocityLagDynamics:
    return VelocityLagDynamics(DynamicsConfig(**overrides))


def test_yaml_config_loads() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = DynamicsConfig.from_dict(yaml.safe_load(f)["dynamics"])
    assert cfg.dt == pytest.approx(0.05)


def test_zero_action_stays_at_rest() -> None:
    d = make()
    d.reset(np.array([1.0, 2.0, 3.0]))
    for _ in range(50):
        d.step(np.zeros(4))
    np.testing.assert_allclose(d.state.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(d.state.velocity, 0.0)


def test_first_step_matches_hand_calculation() -> None:
    # Hand-computed in the tutorial: alpha = 1 - exp(-0.05/0.3) = 0.15352
    d = make(max_speed_xy=2.0, max_accel=1e9)
    d.reset(np.zeros(3), yaw=0.0)
    d.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert d.state.velocity[0] == pytest.approx(0.30704, abs=1e-4)
    d.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert d.state.velocity[0] == pytest.approx(0.56694, abs=1e-4)


def test_velocity_converges_to_command() -> None:
    d = make()
    d.reset(np.zeros(3))
    for _ in range(200):
        d.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert d.state.velocity[0] == pytest.approx(3.0, rel=0.01)


def test_acceleration_is_limited() -> None:
    d = make(max_accel=2.0)
    d.reset(np.zeros(3))
    prev = d.state.velocity.copy()
    for _ in range(40):
        d.step(np.array([1.0, 1.0, 1.0, 0.0]))
        accel = np.linalg.norm(d.state.velocity - prev) / d.config.dt
        assert accel <= 2.0 + 1e-9
        prev = d.state.velocity.copy()


def test_forward_follows_body_frame() -> None:
    # Facing +y (yaw = 90 degrees): "forward" must move the drone along world +y.
    d = make()
    d.reset(np.zeros(3), yaw=np.pi / 2)
    for _ in range(40):
        d.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert d.state.position[1] > 1.0
    assert abs(d.state.position[0]) < 1e-6


def test_yaw_stays_wrapped() -> None:
    d = make()
    d.reset(np.zeros(3))
    for _ in range(500):
        d.step(np.array([0.0, 0.0, 0.0, 1.0]))
        assert -np.pi <= d.state.yaw < np.pi


def test_out_of_range_actions_are_clipped() -> None:
    a, b = make(), make()
    a.reset(np.zeros(3))
    b.reset(np.zeros(3))
    for _ in range(10):
        a.step(np.array([10.0, -10.0, 10.0, 10.0]))
        b.step(np.array([1.0, -1.0, 1.0, 1.0]))
    np.testing.assert_allclose(a.state.position, b.state.position)