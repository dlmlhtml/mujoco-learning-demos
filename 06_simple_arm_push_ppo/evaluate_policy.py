# -*- coding: utf-8 -*-
"""Evaluate the PPO policy with success rate and distance measurements."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from simple_arm_push_env import SimpleArmPushEnv


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ppo_simple_arm_push.zip"
EPISODES = 10


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model not found: {MODEL_PATH}. Run python train_ppo.py first."
        )

    model = PPO.load(str(MODEL_PATH))
    successes = 0
    final_distances = []
    minimum_distances = []

    for seed in range(EPISODES):
        env = SimpleArmPushEnv()
        obs, info = env.reset(seed=seed)
        minimum_distance = info["cube_to_goal"]

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            minimum_distance = min(minimum_distance, info["cube_to_goal"])
            if terminated or truncated:
                break

        successes += int(info["is_success"])
        final_distances.append(info["cube_to_goal"])
        minimum_distances.append(minimum_distance)
        env.close()

    print(f"success rate: {successes}/{EPISODES}")
    print("final distance:", np.round(final_distances, 3))
    print("minimum distance:", np.round(minimum_distances, 3))


if __name__ == "__main__":
    main()
