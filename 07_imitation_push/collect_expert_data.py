# -*- coding: utf-8 -*-
"""运行 scripted expert，并按 episode 保存机器人 demonstration dataset。"""

import argparse
import json
from pathlib import Path

import numpy as np

from imitation_push_env import ImitationPushEnv
from scripted_expert import INSTRUCTION, scripted_expert_action


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    return parser.parse_args()


def collect_episode(env, seed):
    """返回一个episode；所有 *_t 数据都在执行 action_t 之前记录。"""
    obs, reset_info = env.reset(seed=seed)
    states = []
    images = []
    actions = []
    next_states = []
    rewards = []
    terminated_flags = []
    truncated_flags = []

    while True:
        # 这是时刻t的观测。专家必须根据同一个state_t生成action_t。
        state_t = obs.copy()
        image_t = env.render().copy()
        action_t = scripted_expert_action(env)

        next_obs, reward, terminated, truncated, info = env.step(action_t)

        states.append(state_t)
        images.append(image_t)
        actions.append(action_t)
        next_states.append(next_obs.copy())
        rewards.append(reward)
        terminated_flags.append(terminated)
        truncated_flags.append(truncated)

        obs = next_obs
        if terminated or truncated:
            break

    arrays = {
        "state": np.asarray(states, dtype=np.float32),
        "image": np.asarray(images, dtype=np.uint8),
        "action": np.asarray(actions, dtype=np.float32),
        "next_state": np.asarray(next_states, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated_flags, dtype=bool),
        "truncated": np.asarray(truncated_flags, dtype=bool),
    }
    metadata = {
        "seed": seed,
        "instruction": INSTRUCTION,
        "length": len(states),
        "success": bool(info["is_success"]),
        "initial_cube_pos": reset_info["cube_pos"].tolist(),
        "goal_pos": reset_info["goal_pos"].tolist(),
        "final_cube_to_goal": float(info["cube_to_goal"]),
    }
    return arrays, metadata


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob("episode_*"))
    if existing:
        raise FileExistsError(
            f"dataset is not empty: {args.output}. Move it or choose --output."
        )

    env = ImitationPushEnv()
    total_steps = 0

    try:
        for episode_index in range(args.episodes):
            arrays, metadata = collect_episode(env, args.seed + episode_index)
            if not metadata["success"]:
                raise RuntimeError(
                    f"expert failed at episode {episode_index}; dataset was stopped"
                )

            episode_dir = args.output / f"episode_{episode_index:04d}"
            episode_dir.mkdir()
            np.savez_compressed(episode_dir / "trajectory.npz", **arrays)
            (episode_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            total_steps += metadata["length"]
            print(
                f"episode {episode_index + 1:03d}/{args.episodes}: "
                f"steps={metadata['length']:3d}, "
                f"distance={metadata['final_cube_to_goal']:.3f}"
            )
    finally:
        env.close()

    print(f"saved {args.episodes} episodes / {total_steps} transitions to: {args.output}")


if __name__ == "__main__":
    main()

