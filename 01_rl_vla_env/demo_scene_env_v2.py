# -*- coding: utf-8 -*-
from pathlib import Path
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np


ROOT = Path(__file__).resolve().parent
ROBOT_XML = ROOT / "robots" / "mini_humanoid_motor.xml"


SCENE_OBJECTS_XML = """
    <!-- 额外光源：让障碍物和目标方块更清楚。 -->
    <light name="side_light" pos="-2 2 3" dir="1 -1 -1" diffuse="0.5 0.5 0.5"/>

    <!-- 固定障碍物：没有 body/freejoint，所以它们固定在世界里。 -->
    <geom name="obstacle_0" type="box" pos="0.65 0.35 0.08"
          size="0.12 0.12 0.08" rgba="0.55 0.30 0.85 1"/>
    <geom name="obstacle_1" type="box" pos="0.85 -0.30 0.06"
          size="0.10 0.18 0.06" rgba="0.15 0.65 0.70 1"/>
    <geom name="obstacle_2" type="cylinder" pos="0.35 -0.55 0.08"
          size="0.10 0.08" rgba="0.95 0.62 0.15 1"/>

    <!-- 可移动目标方块：freejoint 让它可以平移和旋转。 -->
    <body name="target_cube" pos="0.55 0 0.12">
      <freejoint name="target_cube_freejoint"/>
      <geom name="target_cube_geom" type="box" size="0.08 0.08 0.08"
            mass="0.2" rgba="0.10 0.78 0.25 1"/>
    </body>

    <!-- 任务目标区域：现在只做可视化标记，不参与碰撞。 -->
    <geom name="goal_marker" type="cylinder" pos="1.05 0 0.01"
          size="0.16 0.01" contype="0" conaffinity="0" rgba="0.1 0.9 0.25 0.35"/>
"""


class MiniHumanoidSceneEnvV2:
    """Gymnasium 风格的 MuJoCo 环境雏形，面向后续 RL/VLA 训练格式。"""

    def __init__(self, seed=0, max_episode_steps=500, image_size=(160, 120)):
        # 随机数发生器：所有 reset 随机化都从这里来，方便复现实验。
        self.rng = np.random.default_rng(seed)

        # max_episode_steps 用来控制一局 episode 的最长步数。
        # 到达这个步数后，truncated=True，表示因为时间上限截断。
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # 渲染图像尺寸。VLA/RL 里经常会把图像作为 observation 的一部分。
        self.image_width, self.image_height = image_size

        # 加载 MJCF，创建 model/data。
        # model 是静态结构，data 是运行时状态。
        self.model = mujoco.MjModel.from_xml_string(self._build_scene_xml())
        self.data = mujoco.MjData(self.model)

        # Renderer 是离屏渲染器，不依赖 viewer 窗口。
        # render() 会用它返回 RGB 图像数组。
        self.renderer = mujoco.Renderer(
            self.model, height=self.image_height, width=self.image_width
        )

        # 缓存常用 id，避免每一步都用名字查。
        self.actuator_table = self._build_actuator_table()
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube"
        )
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )

        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_freejoint"
        )
        self.cube_qpos_id = self.model.jnt_qposadr[cube_joint_id]
        self.cube_dof_id = self.model.jnt_dofadr[cube_joint_id]

    def _build_scene_xml(self):
        """把场景物体插入到机器人 MJCF 的 worldbody 里。"""
        robot_xml = ROBOT_XML.read_text(encoding="utf-8")
        insert_at = robot_xml.index("</worldbody>")
        return robot_xml[:insert_at] + SCENE_OBJECTS_XML + robot_xml[insert_at:]

    def _build_actuator_table(self):
        """整理 actuator -> joint -> qpos/qvel 的映射表。"""
        table = []

        for actuator_id in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            table.append(
                {
                    "actuator_id": actuator_id,
                    "actuator_name": actuator_name,
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "qpos_id": self.model.jnt_qposadr[joint_id],
                    "dof_id": self.model.jnt_dofadr[joint_id],
                    "range": self.model.jnt_range[joint_id].copy(),
                }
            )

        return table

    def reset(self):
        """标准 reset 接口：初始化环境，返回 obs 和 info。"""
        # 1. 重置 MuJoCo 运行时状态。
        #    时间、qpos、qvel、ctrl、接触缓存等都会回到默认。
        mujoco.mj_resetData(self.model, self.data)

        # 2. 重置 episode 内部计数。
        self.step_count = 0

        # 3. 随机化目标方块初始位置。
        #    这就是训练环境里的 domain/task randomization 雏形。
        cube_xy = self.rng.uniform(low=[0.35, -0.35], high=[0.85, 0.35])

        # freejoint 的 qpos = [x, y, z, qw, qx, qy, qz]。
        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 3] = [
            cube_xy[0],
            cube_xy[1],
            0.16,
        ]
        self.data.qpos[self.cube_qpos_id + 3 : self.cube_qpos_id + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        # 4. 清空 cube 的速度。
        #    freejoint 的 qvel = [vx, vy, vz, wx, wy, wz]。
        self.data.qvel[self.cube_dof_id : self.cube_dof_id + 6] = 0.0

        # 5. 手动改 qpos/qvel 后，用 mj_forward 刷新 xpos/xquat 等派生数据。
        mujoco.mj_forward(self.model, self.data)

        obs = self.get_observation()
        info = self.get_info()
        return obs, info

    def step(self, action):
        """标准 step 接口：执行 action，返回 obs/reward/done/info。"""
        # 1. action 是外部策略给环境的动作。
        #    当前模型有 8 个 motor，所以 action shape 应该是 (8,)。
        action = np.asarray(action, dtype=float)
        if action.shape != (self.model.nu,):
            raise ValueError(f"action shape must be {(self.model.nu,)}, got {action.shape}")

        # 2. 限制电机命令，防止过大 action 破坏仿真稳定性。
        action = np.clip(action, -12.0, 12.0)

        # 3. 写入 MuJoCo actuator 控制通道。
        #    这行就是 RL/VLA action 真正进入物理仿真的入口。
        self.data.ctrl[:] = action

        # 4. 推进一个物理步。
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # 5. 计算环境返回值。
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = self.is_success()
        truncated = self.step_count >= self.max_episode_steps
        info = self.get_info()

        return obs, reward, terminated, truncated, info

    def get_observation(self):
        """返回 dict observation：state 给控制/RL，image 给后续 VLA。"""
        torso_pos = self.data.xpos[self.torso_body_id]
        torso_quat = self.data.xquat[self.torso_body_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        cube_quat = self.data.xquat[self.cube_body_id]

        # state 是低维状态向量。
        # 这里包含 qpos/qvel、躯干位姿、方块位姿。
        state = np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                torso_pos.copy(),
                torso_quat.copy(),
                cube_pos.copy(),
                cube_quat.copy(),
            ]
        )

        # VLA 常用图像 observation，所以这里直接返回 image。
        # 如果只做普通 RL，也可以暂时不用 image。
        return {
            "state": state,
            "image": self.render(),
        }

    def get_info(self):
        """返回调试信息。info 不参与训练核心状态，但方便日志和排错。"""
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        goal_pos = self.model.geom_pos[self.goal_geom_id].copy()
        distance_to_goal = float(np.linalg.norm(cube_pos[:2] - goal_pos[:2]))

        return {
            "time": float(self.data.time),
            "step_count": self.step_count,
            "cube_pos": cube_pos,
            "goal_pos": goal_pos,
            "distance_to_goal": distance_to_goal,
            "is_success": self.is_success(),
        }

    def compute_reward(self):
        """简单 reward：方块越靠近绿色目标区域，奖励越高。"""
        cube_pos = self.data.xpos[self.cube_body_id]
        goal_pos = self.model.geom_pos[self.goal_geom_id]
        distance = np.linalg.norm(cube_pos[:2] - goal_pos[:2])

        # 距离惩罚：越远 reward 越低。
        reward = -float(distance)

        # 成功奖励：进入目标半径后给额外奖励。
        if distance < 0.12:
            reward += 1.0

        return reward

    def is_success(self):
        """任务成功条件：cube 的 x/y 位置进入目标区域。"""
        cube_pos = self.data.xpos[self.cube_body_id]
        goal_pos = self.model.geom_pos[self.goal_geom_id]
        return bool(np.linalg.norm(cube_pos[:2] - goal_pos[:2]) < 0.12)

    def render(self):
        """离屏渲染，返回 RGB 图像数组，shape=(H, W, 3)。"""
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def neutral_action(self):
        """零 action：所有电机输入为 0。"""
        return np.zeros(self.model.nu)

    def demo_action(self):
        """演示动作：用简单 PD 产生一组周期性动作。"""
        t = self.data.time
        wave = 0.5 + 0.5 * np.sin(t * 1.5)
        action = np.zeros(self.model.nu)

        for item in self.actuator_table:
            name = item["joint_name"]
            q = self.data.qpos[item["qpos_id"]]
            dq = self.data.qvel[item["dof_id"]]
            lower, upper = item["range"]

            target = 0.0
            if name == "left_shoulder_pitch":
                target = -0.7 * wave
            elif name == "right_shoulder_pitch":
                target = 0.7 * wave
            elif "knee" in name:
                target = 0.25 + 0.65 * wave
            elif "ankle" in name:
                target = -0.25 * wave

            target = np.clip(target, lower, upper)
            action[item["actuator_id"]] = 16.0 * (target - q) - 2.2 * dq

        return np.clip(action, -12.0, 12.0)

    def close(self):
        """释放渲染资源。长时间训练或反复创建环境时要调用。"""
        self.renderer.close()


def main():
    env = MiniHumanoidSceneEnvV2(seed=7, max_episode_steps=500)
    obs, info = env.reset()

    print("v2 env ready")
    print("reset returns: obs, info")
    print("step returns: obs, reward, terminated, truncated, info")
    print("action shape:", (env.model.nu,))
    print("state shape:", obs["state"].shape)
    print("image shape:", obs["image"].shape)
    print("initial info:", info)
    print("-" * 70)

    with viewer.launch_passive(env.model, env.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0.35, 0.0, 0.65]
        v.cam.distance = 3.0
        v.cam.azimuth = 135
        v.cam.elevation = -20

        while v.is_running():
            action = env.demo_action()
            obs, reward, terminated, truncated, info = env.step(action)

            if info["step_count"] % 100 == 0:
                print("time:", round(info["time"], 2))
                print("reward:", round(reward, 3))
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("distance_to_goal:", round(info["distance_to_goal"], 3))
                print("state shape:", obs["state"].shape)
                print("image shape:", obs["image"].shape)
                print("-" * 70)

            if terminated or truncated:
                obs, info = env.reset()
                print("episode reset")

            v.sync()
            time.sleep(env.model.opt.timestep)

    env.close()


if __name__ == "__main__":
    main()
