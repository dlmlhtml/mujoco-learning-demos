# -*- coding: utf-8 -*-
"""在未见过的随机目标上做闭环rollout，检验BC是否真的能完成任务。"""

import argparse
from pathlib import Path

import numpy as np

from bc_model import load_checkpoint, predict_action
from imitation_push_env import ImitationPushEnv


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "bc_push.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    model, state_mean, state_std = load_checkpoint(args.model)
    env = ImitationPushEnv()
    successes = 0
    episode_steps = []
    final_distances = []

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            while True:
                action = predict_action(model, state_mean, state_std, obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            successes += int(info["is_success"])
            episode_steps.append(info["step_count"])
            final_distances.append(info["cube_to_goal"])
    finally:
        env.close()

    print(f"BC success rate: {successes}/{args.episodes}")
    print(f"mean episode steps: {np.mean(episode_steps):.1f}")
    print(f"mean final distance: {np.mean(final_distances):.3f}")
    print(f"worst final distance: {np.max(final_distances):.3f}")


if __name__ == "__main__":
    main()

