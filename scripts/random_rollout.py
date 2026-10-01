"""Run one episode with a random or a simple scripted policy and save a trajectory plot."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from uav_nav.envs.config import load_env_config
from uav_nav.envs.nav_env import UavNavEnv
from uav_nav.visualization.episode_plot import plot_episode

logger = logging.getLogger("random_rollout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obstacles", type=int, default=0, help="number of obstacles")
    parser.add_argument("--policy", choices=["random", "scripted"], default="random")
    parser.add_argument("--out", default="results/rollout.png")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_env_config(overrides={"obstacles": {"num_min": args.obstacles, "num_max": args.obstacles}})
    env = UavNavEnv(cfg)
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    diag = float(np.linalg.norm(env.arena))

    total, done = 0.0, False
    while not done:
        if args.policy == "random":
            action = rng.uniform(-1, 1, size=4)
        else:  # fly straight at the goal with a proportional controller (ignores obstacles)
            action = np.append(np.clip(obs[:3] * diag * 0.4, -1, 1), 0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated

    logger.info("Ended by: %s after %d steps, total reward %.2f", info["termination"], info["steps"], total)
    logger.info("Saved plot to %s", plot_episode(env, args.out, f"{args.policy} policy, seed {args.seed}"))


if __name__ == "__main__":
    main()