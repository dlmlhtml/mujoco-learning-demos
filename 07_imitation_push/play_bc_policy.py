# -*- coding: utf-8 -*-
"""加载BC模型，在MuJoCo交互窗口中连续回放随机目标任务。"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer

from bc_model import load_checkpoint, predict_action
from imitation_push_env import ImitationPushEnv


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bc_push.pt"


def main():
    model, state_mean, state_std = load_checkpoint(MODEL_PATH)
    env = ImitationPushEnv()
    episode = 0
    obs, _ = env.reset(seed=10000 + episode)

    with viewer.launch_passive(env.model, env.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0.42, 0.0, 0.18]
        v.cam.distance = 1.25
        v.cam.azimuth = 135
        v.cam.elevation = -35

        while v.is_running():
            action = predict_action(model, state_mean, state_std, obs)
            obs, _, terminated, truncated, info = env.step(action)
            v.sync()
            time.sleep(env.model.opt.timestep * env.config.frame_skip)

            if terminated or truncated:
                print(
                    f"episode={episode}, success={info['is_success']}, "
                    f"steps={info['step_count']}, distance={info['cube_to_goal']:.3f}"
                )
                time.sleep(0.5)
                episode += 1
                obs, _ = env.reset(seed=10000 + episode)

    env.close()


if __name__ == "__main__":
    main()

