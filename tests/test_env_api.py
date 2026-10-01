"""API, determinism and termination tests for UavNavEnv."""

import numpy as np
from stable_baselines3.common.env_checker import check_env

from uav_nav.envs.config import load_env_config
from uav_nav.envs.nav_env import UavNavEnv


def test_check_env() -> None:
    check_env(UavNavEnv(), warn=True)


def test_spaces_and_obs_shape() -> None:
    env = UavNavEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (34,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (4,)
    assert "distance_to_goal" in info


def _rollout(seed: int) -> np.ndarray:
    cfg = load_env_config(overrides={"obstacles": {"num_min": 3, "num_max": 3}})
    env = UavNavEnv(cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(123)
    frames = [obs]
    for _ in range(50):
        obs, *_ = env.step(rng.uniform(-1, 1, 4))
        frames.append(obs)
    return np.array(frames)


def test_same_seed_is_deterministic() -> None:
    np.testing.assert_array_equal(_rollout(7), _rollout(7))


def test_different_seeds_differ() -> None:
    assert not np.array_equal(_rollout(1), _rollout(2))


def test_reset_options_are_respected() -> None:
    env = UavNavEnv()
    env.reset(options={"start": [2, 3, 4], "goal": [15, 15, 5], "yaw": 0.0, "obstacles": [[8, 8, 5, 1.0]]})
    np.testing.assert_allclose(env.dynamics.state.position, [2, 3, 4])
    np.testing.assert_allclose(env.goal, [15, 15, 5])
    assert env.obstacles.shape == (1, 4)


def test_collision_terminates() -> None:
    env = UavNavEnv()
    env.reset(options={"start": [5, 10, 5], "goal": [15, 10, 5], "yaw": 0.0, "obstacles": [[8, 10, 5, 1.0]]})
    info = {}
    terminated = False
    for _ in range(200):
        _, reward, terminated, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
        if terminated:
            break
    assert terminated and info["termination"] == "collision"
    assert reward < -5.0


def test_out_of_bounds_terminates() -> None:
    env = UavNavEnv()
    env.reset(options={"start": [1, 10, 5], "goal": [15, 10, 5], "yaw": 0.0})
    info = {}
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
        if terminated:
            break
    assert info["termination"] == "out_of_bounds"


def test_timeout_truncates() -> None:
    env = UavNavEnv(load_env_config(overrides={"task": {"max_steps": 5}}))
    env.reset(seed=0)
    for i in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert truncated and not terminated and info["termination"] == "timeout"


def test_scripted_controller_reaches_goal() -> None:
    env = UavNavEnv()
    diag = float(np.linalg.norm(env.arena))
    for seed in range(5):
        obs, _ = env.reset(seed=seed, options={"yaw": 0.0})
        info = {}
        for _ in range(600):
            action = np.append(np.clip(obs[:3] * diag * 0.4, -1, 1), 0.0)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert info["termination"] == "goal", f"seed {seed}: {info}"


def test_observation_stays_in_bounds() -> None:
    cfg = load_env_config(overrides={"obstacles": {"num_min": 5, "num_max": 5}})
    env = UavNavEnv(cfg)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        obs, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 4))
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()


def test_sampled_obstacles_keep_spawn_clear() -> None:
    cfg = load_env_config(overrides={"obstacles": {"num_min": 6, "num_max": 6}})
    env = UavNavEnv(cfg)
    min_clear = cfg["task"]["spawn_clearance"] - 1e-9
    for seed in range(20):
        env.reset(seed=seed)
        start = env.dynamics.state.position
        for ox, oy, oz, r in env.obstacles:
            c = np.array([ox, oy, oz])
            assert np.linalg.norm(c - start) - r >= min_clear
            assert np.linalg.norm(c - env.goal) - r >= min_clear