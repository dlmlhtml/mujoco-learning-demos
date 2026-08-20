# -*- coding: utf-8 -*-
"""
加载 PPO 模型并在 MuJoCo viewer 里回放。

运行：
    cd C:\\Users\\Administrator\\Desktop\\robo\\06_simple_arm_push_ppo
    python play_policy.py
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np
from stable_baselines3 import PPO

from simple_arm_push_env import SimpleArmPushEnv


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ppo_simple_arm_push.zip"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model not found: {MODEL_PATH}. Run python train_ppo.py first."
        )

    env = SimpleArmPushEnv()
    model = PPO.load(str(MODEL_PATH))
    obs, info = env.reset(seed=0)

    print("Play PPO policy")
    print("initial info:", info)
    print("-" * 80)

    with viewer.launch_passive(env.model, env.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0.38, 0.00, 0.22]
        v.cam.distance = 1.35
        v.cam.azimuth = 135
        v.cam.elevation = -30

        while v.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if info["step_count"] % 50 == 0:
                print("step:", info["step_count"])
                print("reward:", round(float(reward), 4))
                print("cube_to_goal:", round(info["cube_to_goal"], 4))
                print("contact:", info["contact"])
                print("success:", info["is_success"])
                print("action:", np.round(action, 3))
                print("-" * 80)

            if terminated or truncated:
                print("episode done:", {"terminated": terminated, "truncated": truncated})
                obs, info = env.reset()

            v.sync()
            time.sleep(env.model.opt.timestep * env.config.frame_skip)

    env.close()


if __name__ == "__main__":
    main()

