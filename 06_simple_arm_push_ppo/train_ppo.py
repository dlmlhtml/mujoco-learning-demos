# -*- coding: utf-8 -*-
"""
训练 PPO。

运行前安装：
    conda activate mujoco_learn
    pip install -r requirements-ppo.txt

训练：
    cd C:\\Users\\Administrator\\Desktop\\robo\\06_simple_arm_push_ppo
    python train_ppo.py

你现在要理解：
    PPO 不知道 IK，不知道 push_dir。
    PPO 只看到 observation，输出 action，然后根据 reward 更新 policy。
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from simple_arm_push_env import SimpleArmPushEnv


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ppo_simple_arm_push"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    env = SimpleArmPushEnv()

    # check_env 会检查 reset/step/action_space/observation_space 是否符合 Gymnasium 规范。
    check_env(env, warn=True)

    # Monitor 会记录 episode reward、length、success 等日志，方便看训练过程。
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        seed=0,
        tensorboard_log=str(ROOT / "runs"),
    )

    # Push 比 reach 难，30k 往往还学不会稳定推。
    # 这里默认跑 100k，目标是让 policy 至少学到“接触后别把方块越推越远”。
    model.learn(total_timesteps=100_000)
    model.save(str(MODEL_PATH))

    env.close()
    print(f"saved model to: {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
