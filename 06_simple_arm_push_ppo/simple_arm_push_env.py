# -*- coding: utf-8 -*-
"""
Gymnasium 风格的 MuJoCo 推方块环境。

这一步和 05_simple_arm_push 最大的区别：

    05:
        人手写 controller，自己决定末端去哪里，再用 IK/PD 推方块。

    06:
        RL policy 输出 action。
        环境只负责接收 action、推进 MuJoCo、返回 obs/reward/done。

所以现在你要盯住 RL 四要素：
    observation: policy 能看到什么
    action:      policy 能控制什么
    reward:      policy 怎么知道好坏
    episode:     一次从 reset 到 done 的尝试
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SUCCESS_RADIUS = 0.16


XML = """
<mujoco model="simple_arm_push_rl">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.02" limited="true"/>
    <geom friction="1.0 0.02 0.001"/>
    <motor ctrllimited="true" ctrlrange="-80 80"/>
  </default>

  <asset>
    <material name="mat_floor" rgba="0.86 0.86 0.84 1"/>
    <material name="mat_base" rgba="0.20 0.25 0.30 1"/>
    <material name="mat_link1" rgba="0.15 0.45 0.95 1"/>
    <material name="mat_link2" rgba="0.95 0.55 0.15 1"/>
    <material name="mat_joint" rgba="0.92 0.10 0.12 1"/>
    <material name="mat_ee" rgba="0.95 0.05 0.05 1"/>
    <material name="mat_cube" rgba="0.10 0.35 0.95 1"/>
    <material name="mat_goal" rgba="0.10 0.85 0.25 0.35"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="1.4 -2.0 1.0" xyaxes="0.82 0.57 0 -0.23 0.34 0.91"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor"/>

    <body name="cube" pos="0.42 -0.18 0.045">
      <freejoint name="cube_freejoint"/>
      <geom name="cube_geom" type="box" size="0.045 0.045 0.045"
            mass="0.12" material="mat_cube"/>
    </body>

    <geom name="goal_marker" type="cylinder" pos="0.68 0.18 0.01"
          size="0.16 0.01" contype="0" conaffinity="0" material="mat_goal"/>

    <body name="base" pos="0 0 0.08">
      <geom name="base_geom" type="cylinder" size="0.12 0.08"
            contype="0" conaffinity="0" material="mat_base"/>

      <body name="yaw_link" pos="0 0 0.08">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom name="base_yaw_marker" type="sphere" size="0.045" mass="0.05"
              contype="0" conaffinity="0" material="mat_joint"/>

        <body name="upper_arm" pos="0 0 0.08">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom name="shoulder_marker" type="sphere" size="0.045" mass="0.05"
                contype="0" conaffinity="0" material="mat_joint"/>
          <geom name="upper_arm_geom" type="capsule"
                fromto="0 0 0 0.35 0 0" size="0.035" mass="0.8"
                contype="0" conaffinity="0" material="mat_link1"/>

          <body name="forearm" pos="0.35 0 0">
            <joint name="elbow_pitch" type="hinge" axis="0 1 0" range="-1.9 1.9"/>
            <geom name="elbow_marker" type="sphere" size="0.04" mass="0.05"
                  contype="0" conaffinity="0" material="mat_joint"/>
            <geom name="forearm_geom" type="capsule"
                  fromto="0 0 0 0.30 0 0" size="0.028" mass="0.5"
                  contype="0" conaffinity="0" material="mat_link2"/>

            <site name="ee_site" pos="0.30 0 0" size="0.025" material="mat_ee"/>
            <geom name="ee_geom" type="sphere" pos="0.30 0 0"
                  size="0.035" mass="0.08" material="mat_ee"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="base_yaw_motor" joint="base_yaw"/>
    <motor name="shoulder_pitch_motor" joint="shoulder_pitch"/>
    <motor name="elbow_pitch_motor" joint="elbow_pitch"/>
  </actuator>
</mujoco>
"""


@dataclass
class PushEnvConfig:
    """环境配置，先集中放这里，后面调参方便。"""

    max_episode_steps: int = 300
    frame_skip: int = 4
    cartesian_delta_scale: float = 0.025
    kp: float = 120.0
    kd: float = 10.0
    # First learn one fixed push task. Turn this on later for generalization.
    randomize_reset: bool = False
    render_width: int = 320
    render_height: int = 240


class SimpleArmPushEnv(gym.Env):
    """
    一个最小 PPO 训练环境。

    Gymnasium 标准接口：
        reset() -> obs, info
        step(action) -> obs, reward, terminated, truncated, info
        render() -> RGB image

    当前 action 设计：
        policy 输出末端执行器的 3 个位移方向 [dx, dy, dz]，范围 [-1, 1]。
        环境先用 Jacobian IK 转成 target_q，再用 PD 写入 data.ctrl。

    这样 PPO 学“末端往哪走”，底层控制器负责“关节怎么配合”。
    链路变成：
        PPO action -> delta xyz -> Jacobian IK -> target_q -> PD -> data.ctrl
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: PushEnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or PushEnvConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self.ee_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ee_geom"
        )
        self.cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )

        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint"
        )
        self.cube_qpos_id = self.model.jnt_qposadr[cube_joint_id]
        self.cube_dof_id = self.model.jnt_dofadr[cube_joint_id]

        self.joint_names = ["base_yaw", "shoulder_pitch", "elbow_pitch"]
        self.qpos_ids = []
        self.dof_ids = []
        self.joint_ranges = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            self.qpos_ids.append(self.model.jnt_qposadr[joint_id])
            self.dof_ids.append(self.model.jnt_dofadr[joint_id])
            self.joint_ranges.append(self.model.jnt_range[joint_id].copy())

        self.qpos_ids = np.array(self.qpos_ids, dtype=int)
        self.dof_ids = np.array(self.dof_ids, dtype=int)
        self.joint_ranges = np.array(self.joint_ranges, dtype=float)

        # action_space 告诉 PPO：输出 3 个连续动作 [dx, dy, dz]。
        # 它们是末端位移方向，不是 3 个 motor 的力矩。
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # observation 是一个 24 维向量：
        #   3 joint qpos
        #   3 joint qvel
        #   3 ee_pos
        #   3 cube_pos
        #   3 goal_pos
        #   3 cube_to_goal
        #   2 push direction (xy)
        #   3 ee-to-behind-target vector
        #   1 contact flag
        self.obs_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.renderer = None
        self.step_count = 0
        self.target_q = np.zeros(self.model.nu, dtype=float)
        self.prev_cube_to_goal = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """开始一个新 episode。"""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # 初始姿态沿用 05 demo：肘部朝下，末端比较容易接近桌面。
        # Curriculum stage 1: start the end effector on the correct side of the cube.
        # These joint angles place ee_site near [0.361, -0.261, 0.075].
        # The policy therefore learns contact + push first, without also solving a
        # collision-free path-planning problem around the cube.
        self.data.qpos[self.qpos_ids] = np.array([-0.625521, -0.326740, 1.506330])
        self.target_q = self.data.qpos[self.qpos_ids].copy()

        # 训练时加入一点随机化，让 policy 不只记住一个固定起点。
        # 但随机范围先别太大，push 任务本来就比 reach 难。
        cube_pos = np.array([0.42, -0.18, 0.045], dtype=float)
        goal_pos = np.array([0.68, 0.18, 0.01], dtype=float)
        if self.config.randomize_reset:
            cube_pos[:2] += self.np_random.uniform([-0.04, -0.04], [0.04, 0.04])
            goal_pos[:2] += self.np_random.uniform([-0.04, -0.04], [0.04, 0.04])

        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 3] = cube_pos
        self.data.qpos[self.cube_qpos_id + 3 : self.cube_qpos_id + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qvel[self.cube_dof_id : self.cube_dof_id + 6] = 0.0

        # goal_marker 是固定 geom，训练时为了随机目标，直接改 model.geom_pos。
        # 因为它不参与碰撞，只是目标标记和成功判断位置。
        self.model.geom_pos[self.goal_geom_id] = goal_pos

        mujoco.mj_forward(self.model, self.data)
        self.prev_cube_to_goal = self._cube_to_goal_distance()
        obs = self.get_observation()
        info = self.get_info()
        return obs, info

    def step(self, action):
        """
        执行 PPO 给出的 action。

        action 是 policy 输出的连续向量，不再由手写 push_dir 生成。
        在这个环境里：
            action -> delta xyz -> Jacobian IK -> target_q -> PD -> data.ctrl -> mj_step
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # PPO 输出归一化的末端位移方向 [-1, 1]。
        # Jacobian IK 把它转成关节目标，PD 再把关节目标转成力矩。
        self._set_joint_target_from_cartesian_action(action)

        for _ in range(self.config.frame_skip):
            self.data.ctrl[:] = self._pd_control()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self.get_observation()
        reward = self._compute_reward(action)
        self.prev_cube_to_goal = self._cube_to_goal_distance()
        terminated = self.is_success()
        truncated = self.step_count >= self.config.max_episode_steps
        info = self.get_info()
        return obs, reward, terminated, truncated, info

    def _pd_control(self):
        """把 PPO 给出的 target_q 转成 MuJoCo motor ctrl。"""
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]
        torque = self.config.kp * (self.target_q - q) - self.config.kd * dq
        return np.clip(torque, -80.0, 80.0)

    def _set_joint_target_from_cartesian_action(self, action):
        """Convert policy delta xyz into joint targets with one Jacobian IK step."""
        desired_delta = action.astype(float) * self.config.cartesian_delta_scale

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        jacobian = jacp[:, self.dof_ids]

        damping = 0.08
        inverse = jacobian.T @ np.linalg.inv(
            jacobian @ jacobian.T + damping * damping * np.eye(3)
        )
        joint_delta = inverse @ desired_delta
        joint_delta = np.clip(joint_delta, -0.15, 0.15)

        current_q = self.data.qpos[self.qpos_ids].copy()
        self.target_q = np.clip(
            current_q + joint_delta,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )

    def get_observation(self):
        """把 MuJoCo 状态整理成 PPO 能吃的一维向量。"""
        q = self.data.qpos[self.qpos_ids].copy()
        dq = self.data.qvel[self.dof_ids].copy()
        ee_pos = self.get_ee_pos()
        cube_pos = self.get_cube_pos()
        goal_pos = self.get_goal_pos()
        cube_to_goal = goal_pos - cube_pos
        push_dir = self.get_push_direction()
        ee_to_behind = self.get_behind_target() - ee_pos
        contact = np.array([float(self.has_ee_cube_contact())], dtype=float)

        obs = np.concatenate(
            [
                q,
                dq,
                ee_pos,
                cube_pos,
                goal_pos,
                cube_to_goal,
                push_dir,
                ee_to_behind,
                contact,
            ]
        )
        return obs.astype(np.float32)

    def get_info(self):
        """info 不参与训练核心，但非常适合调试和日志。"""
        cube_to_goal = float(
            np.linalg.norm(self.get_cube_pos()[:2] - self.get_goal_pos()[:2])
        )
        ee_to_cube = float(np.linalg.norm(self.get_ee_pos() - self.get_cube_pos()))
        return {
            "step_count": self.step_count,
            "cube_pos": self.get_cube_pos(),
            "goal_pos": self.get_goal_pos(),
            "ee_pos": self.get_ee_pos(),
            "cube_to_goal": cube_to_goal,
            "ee_to_cube": ee_to_cube,
            "contact": self.has_ee_cube_contact(),
            "is_success": self.is_success(),
        }

    def _compute_reward(self, action):
        """
        PPO 的训练信号。

        主目标：
            方块越靠近目标，reward 越高。

        辅助 shaping：
            末端靠近方块、发生接触、动作别太大。

        这是教学版 reward，不是最终工程最优 reward。
        """
        cube_to_goal = self._cube_to_goal_distance()
        ee_to_cube = float(np.linalg.norm(self.get_ee_pos() - self.get_cube_pos()))
        action_cost = float(np.sum(np.square(action)))
        progress = self.prev_cube_to_goal - cube_to_goal
        behind_distance = float(
            np.linalg.norm(self.get_ee_pos() - self.get_behind_target())
        )
        push_dir = self.get_push_direction()
        ee_from_cube = self.get_ee_pos()[:2] - self.get_cube_pos()[:2]
        signed_side = float(np.dot(ee_from_cube, push_dir))

        # 主奖励仍然是方块靠近目标。
        reward = -0.25 * cube_to_goal

        # progress 是关键修正：
        #   方块这一帧更靠近目标 -> 正奖励
        #   方块这一帧远离目标   -> 负奖励
        # 这能直接惩罚“到了方块后面但越推越远”的行为。
        # Cube motion is millimetres per step, so progress needs a visible scale.
        # Positive means moving toward the goal; negative means pushing away.
        reward += 200.0 * progress

        # Dense guidance for the hard part of pushing: approach from the correct side.
        # A small penalty keeps the end effector near the cube. The dominant signal
        # remains actual cube progress, so hovering near the cube is not enough.
        reward -= 0.03 * behind_distance
        reward -= 0.03 * ee_to_cube
        reward -= 0.20 * max(signed_side, 0.0)
        reward -= 0.01 * action_cost

        if self.has_ee_cube_contact():
            reward += 0.30 if signed_side < 0.0 else -0.50
        if self.is_success():
            reward += 10.0

        return float(reward)

    def _cube_to_goal_distance(self):
        """方块中心到目标中心的 xy 平面距离。"""
        return float(np.linalg.norm(self.get_cube_pos()[:2] - self.get_goal_pos()[:2]))

    def get_push_direction(self):
        """Return the normalized xy direction from the cube to the goal."""
        direction = self.get_goal_pos()[:2] - self.get_cube_pos()[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return np.array([1.0, 0.0], dtype=float)
        return direction / norm

    def get_behind_target(self):
        """Return the point where the end effector should approach before pushing."""
        target = self.get_cube_pos().copy()
        target[:2] -= self.get_push_direction() * 0.10
        target[2] = 0.075
        return target

    def is_success(self):
        """成功条件：方块中心进入目标区域半径。"""
        cube_xy = self.get_cube_pos()[:2]
        goal_xy = self.get_goal_pos()[:2]
        return bool(np.linalg.norm(cube_xy - goal_xy) < SUCCESS_RADIUS)

    def has_ee_cube_contact(self):
        """判断当前帧红色末端球是否接触蓝色方块。"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom_pair = {contact.geom1, contact.geom2}
            if geom_pair == {self.ee_geom_id, self.cube_geom_id}:
                return True
        return False

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    def get_goal_pos(self):
        return self.model.geom_pos[self.goal_geom_id].copy()

    def render(self):
        """返回 RGB 图像，给 play_policy 或后续 VLA 观察用。"""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.config.render_height,
                width=self.config.render_width,
            )
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_env():
    """给 Stable-Baselines3 的简单工厂函数。"""
    return SimpleArmPushEnv()
