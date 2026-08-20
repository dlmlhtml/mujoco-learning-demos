# -*- coding: utf-8 -*-
"""随机目标推块环境：用于收集 demonstration 和训练 Behavior Cloning。"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SUCCESS_RADIUS = 0.16
CUBE_START = np.array([0.42, 0.0, 0.045], dtype=float)


XML = """
<mujoco model="imitation_push">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.02" limited="true"/>
    <geom friction="1.0 0.02 0.001"/>
    <motor ctrllimited="true" ctrlrange="-80 80"/>
  </default>

  <asset>
    <material name="floor" rgba="0.86 0.86 0.84 1"/>
    <material name="base" rgba="0.20 0.25 0.30 1"/>
    <material name="link1" rgba="0.15 0.45 0.95 1"/>
    <material name="link2" rgba="0.95 0.55 0.15 1"/>
    <material name="joint" rgba="0.92 0.10 0.12 1"/>
    <material name="ee" rgba="0.95 0.05 0.05 1"/>
    <material name="cube" rgba="0.10 0.35 0.95 1"/>
    <material name="goal" rgba="0.10 0.85 0.25 0.35"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="1.25 -1.55 1.05"
            xyaxes="0.78 0.63 0 -0.35 0.43 0.84"/>
    <geom name="floor_geom" type="plane" size="2 2 0.1" material="floor"/>

    <body name="cube_body" pos="0.42 0 0.045">
      <freejoint name="cube_freejoint"/>
      <geom name="cube_geom" type="box" size="0.045 0.045 0.045"
            mass="0.12" material="cube"/>
    </body>

    <geom name="goal_marker" type="cylinder" pos="0.70 0 0.01"
          size="0.16 0.01" contype="0" conaffinity="0" material="goal"/>

    <body name="base_body" pos="0 0 0.08">
      <geom name="base_geom" type="cylinder" size="0.12 0.08"
            contype="0" conaffinity="0" material="base"/>
      <body name="yaw_link" pos="0 0 0.08">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom type="sphere" size="0.045" mass="0.05"
              contype="0" conaffinity="0" material="joint"/>
        <body name="upper_arm" pos="0 0 0.08">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="sphere" size="0.045" mass="0.05"
                contype="0" conaffinity="0" material="joint"/>
          <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.035" mass="0.8"
                contype="0" conaffinity="0" material="link1"/>
          <body name="forearm" pos="0.35 0 0">
            <joint name="elbow_pitch" type="hinge" axis="0 1 0" range="-1.9 1.9"/>
            <geom type="sphere" size="0.04" mass="0.05"
                  contype="0" conaffinity="0" material="joint"/>
            <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.028" mass="0.5"
                  contype="0" conaffinity="0" material="link2"/>
            <site name="ee_site" pos="0.30 0 0" size="0.025" material="ee"/>
            <!-- 教学环境只让末端球推动方块，避免连杆提前撞歪方块。 -->
            <geom name="ee_geom" type="sphere" pos="0.30 0 0"
                  size="0.035" mass="0.08" material="ee"/>
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
class EnvConfig:
    max_episode_steps: int = 300
    frame_skip: int = 4
    cartesian_delta_scale: float = 0.025
    kp: float = 120.0
    kd: float = 10.0
    goal_angle_min_deg: float = -70.0
    goal_angle_max_deg: float = 70.0
    # 目标距离受这条0.65 m教学机械臂的可达范围限制。
    goal_distance_min: float = 0.28
    goal_distance_max: float = 0.30
    cube_jitter: float = 0.012
    render_width: int = 96
    render_height: int = 96


class ImitationPushEnv(gym.Env):
    """随机任务环境，action 是归一化末端位移方向 [dx, dy, dz]。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.ee_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "ee_geom")
        self.cube_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        self.cube_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.goal_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")

        cube_joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self.cube_qpos_id = self.model.jnt_qposadr[cube_joint_id]
        self.cube_dof_id = self.model.jnt_dofadr[cube_joint_id]

        self.joint_names = ["base_yaw", "shoulder_pitch", "elbow_pitch"]
        self.qpos_ids = []
        self.dof_ids = []
        self.joint_ranges = []
        for name in self.joint_names:
            joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qpos_ids.append(self.model.jnt_qposadr[joint_id])
            self.dof_ids.append(self.model.jnt_dofadr[joint_id])
            self.joint_ranges.append(self.model.jnt_range[joint_id].copy())
        self.qpos_ids = np.asarray(self.qpos_ids, dtype=int)
        self.dof_ids = np.asarray(self.dof_ids, dtype=int)
        self.joint_ranges = np.asarray(self.joint_ranges, dtype=float)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        # 19维状态不直接包含专家算好的push_dir：
        # q(3), dq(3), ee(3), cube(3), goal(3), goal-cube(3), contact(1)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(19,), dtype=np.float32
        )

        self.target_q = np.zeros(3, dtype=float)
        self.prev_cube_to_goal = 0.0
        self.step_count = 0
        self.renderer = None

    def _name2id(self, object_type, name):
        return mujoco.mj_name2id(self.model, object_type, name)

    def reset(self, *, seed=None, options=None):
        """随机目标方向，并把末端运动学初始化到方块正确推送侧。"""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        cube_pos = CUBE_START.copy()
        cube_pos[:2] += self.np_random.uniform(
            -self.config.cube_jitter, self.config.cube_jitter, size=2
        )
        angle = np.deg2rad(
            self.np_random.uniform(
                self.config.goal_angle_min_deg, self.config.goal_angle_max_deg
            )
        )
        distance = self.np_random.uniform(
            self.config.goal_distance_min, self.config.goal_distance_max
        )
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        goal_pos = np.array(
            [
                cube_pos[0] + direction[0] * distance,
                cube_pos[1] + direction[1] * distance,
                0.01,
            ],
            dtype=float,
        )

        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 3] = cube_pos
        self.data.qpos[self.cube_qpos_id + 3 : self.cube_qpos_id + 7] = [1, 0, 0, 0]
        self.data.qvel[self.cube_dof_id : self.cube_dof_id + 6] = 0.0
        self.model.geom_pos[self.goal_geom_id] = goal_pos

        # 先给IK一个接近目标的姿态种子，再只做运动学迭代，不推进物理时间。
        behind = self.get_behind_target_from(cube_pos, goal_pos)
        yaw_seed = np.arctan2(behind[1], behind[0])
        self.data.qpos[self.qpos_ids] = [yaw_seed, -0.33, 1.50]
        mujoco.mj_forward(self.model, self.data)
        self._solve_reset_ik(behind)

        self.data.qvel[self.dof_ids] = 0.0
        self.target_q = self.data.qpos[self.qpos_ids].copy()
        mujoco.mj_forward(self.model, self.data)
        self.prev_cube_to_goal = self.cube_to_goal_distance()

        return self.get_observation(), self.get_info()

    def _solve_reset_ik(self, target):
        """只在reset使用：通过多次Jacobian迭代把末端放到指定起点。"""
        for _ in range(80):
            error = target - self.get_ee_pos()
            if np.linalg.norm(error) < 1e-5:
                break
            jacobian = self._position_jacobian()
            inverse = self._damped_inverse(jacobian, damping=0.03)
            joint_delta = np.clip(inverse @ error, -0.08, 0.08)
            q = self.data.qpos[self.qpos_ids] + joint_delta
            self.data.qpos[self.qpos_ids] = np.clip(
                q, self.joint_ranges[:, 0], self.joint_ranges[:, 1]
            )
            mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        """执行一次高层action，其中包含4次PD与MuJoCo物理步。"""
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self._set_joint_target(action)

        for _ in range(self.config.frame_skip):
            self.data.ctrl[:] = self._pd_torque()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        current_distance = self.cube_to_goal_distance()
        progress = self.prev_cube_to_goal - current_distance
        reward = 200.0 * progress - 0.25 * current_distance
        if self.is_success():
            reward += 10.0
        self.prev_cube_to_goal = current_distance

        terminated = self.is_success()
        truncated = self.step_count >= self.config.max_episode_steps
        return self.get_observation(), float(reward), terminated, truncated, self.get_info()

    def _set_joint_target(self, action):
        desired_delta = action.astype(float) * self.config.cartesian_delta_scale
        inverse = self._damped_inverse(self._position_jacobian(), damping=0.08)
        joint_delta = np.clip(inverse @ desired_delta, -0.15, 0.15)
        current_q = self.data.qpos[self.qpos_ids]
        self.target_q = np.clip(
            current_q + joint_delta,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )

    def _position_jacobian(self):
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:, self.dof_ids]

    @staticmethod
    def _damped_inverse(jacobian, damping):
        return jacobian.T @ np.linalg.inv(
            jacobian @ jacobian.T + damping * damping * np.eye(3)
        )

    def _pd_torque(self):
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]
        torque = self.config.kp * (self.target_q - q) - self.config.kd * dq
        return np.clip(torque, -80.0, 80.0)

    def get_observation(self):
        cube = self.get_cube_pos()
        goal = self.get_goal_pos()
        return np.concatenate(
            [
                self.data.qpos[self.qpos_ids],
                self.data.qvel[self.dof_ids],
                self.get_ee_pos(),
                cube,
                goal,
                goal - cube,
                [float(self.has_ee_cube_contact())],
            ]
        ).astype(np.float32)

    def get_info(self):
        return {
            "step_count": self.step_count,
            "cube_pos": self.get_cube_pos(),
            "goal_pos": self.get_goal_pos(),
            "ee_pos": self.get_ee_pos(),
            "cube_to_goal": self.cube_to_goal_distance(),
            "contact": self.has_ee_cube_contact(),
            "is_success": self.is_success(),
        }

    def get_push_direction(self):
        direction = self.get_goal_pos()[:2] - self.get_cube_pos()[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.array([1.0, 0.0], dtype=float)
        return direction / norm

    @staticmethod
    def get_behind_target_from(cube_pos, goal_pos):
        direction = goal_pos[:2] - cube_pos[:2]
        direction /= max(np.linalg.norm(direction), 1e-8)
        target = cube_pos.copy()
        target[:2] -= direction * 0.10
        target[2] = 0.075
        return target

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    def get_goal_pos(self):
        return self.model.geom_pos[self.goal_geom_id].copy()

    def cube_to_goal_distance(self):
        return float(np.linalg.norm(self.get_cube_pos()[:2] - self.get_goal_pos()[:2]))

    def is_success(self):
        return self.cube_to_goal_distance() < SUCCESS_RADIUS

    def has_ee_cube_contact(self):
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {contact.geom1, contact.geom2} == {self.ee_geom_id, self.cube_geom_id}:
                return True
        return False

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                width=self.config.render_width,
                height=self.config.render_height,
            )
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
