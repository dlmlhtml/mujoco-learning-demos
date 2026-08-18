# -*- coding: utf-8 -*-
"""
简化机械臂 IK 教学 Demo。

你现在要吃透的主线：
    joint space:      qpos = [关节1角度, 关节2角度, 关节3角度]
    Cartesian space:  ee_pos = [末端 x, 末端 y, 末端 z]

    qpos -> ee_pos            叫 FK / 正向运动学
    target_pos -> target_qpos 叫 IK / 逆向运动学

运行：
    cd C:\\Users\\Administrator\\Desktop\\robo\\04_simple_arm_ik
    python demo_simple_arm_ik.py

指定目标点：
    python demo_simple_arm_ik.py --target 0.55 0.20 0.35
"""

import argparse
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np


# 默认目标点。你可以先直接改这里，看机械臂末端追不同位置。
DEFAULT_TARGET_POS = np.array([0.55, 0.20, 0.35], dtype=float)


# 一个 3 自由度简化机械臂：
# - base_yaw：底座绕 z 轴转，决定朝向左/右。
# - shoulder_pitch：肩关节绕 y 轴转，抬高手臂。
# - elbow_pitch：肘关节绕 y 轴转，弯曲手臂。
#
# 注意：这里故意不用真实 Franka/UR5 模型，先把 FK/IK/Jacobian 主线学清楚。
XML = """
<mujoco model="simple_arm_ik">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.02" limited="true"/>
    <geom friction="0.8 0.02 0.001"/>
    <motor ctrllimited="true" ctrlrange="-40 40"/>
  </default>

  <asset>
    <material name="mat_floor" rgba="0.88 0.88 0.86 1"/>
    <material name="mat_base" rgba="0.20 0.25 0.30 1"/>
    <material name="mat_link1" rgba="0.15 0.45 0.95 1"/>
    <material name="mat_link2" rgba="0.95 0.55 0.15 1"/>
    <material name="mat_joint" rgba="0.92 0.10 0.12 1"/>
    <material name="mat_target" rgba="0.10 0.85 0.25 0.75"/>
    <material name="mat_ee" rgba="0.95 0.05 0.05 1"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="1.5 -2.0 1.1" xyaxes="0.82 0.57 0 -0.25 0.36 0.90"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor"/>

    <!-- 绿色目标球。mocap=true 表示它的位置可以由 Python 直接改 data.mocap_pos。 -->
    <body name="target" mocap="true" pos="0.55 0.20 0.35">
      <geom name="target_geom" type="sphere" size="0.035"
            contype="0" conaffinity="0" material="mat_target"/>
    </body>

    <!-- 机械臂固定在世界中。 -->
    <body name="base" pos="0 0 0.08">
      <geom name="base_geom" type="cylinder" size="0.12 0.08" material="mat_base"/>

      <!-- 第 1 个关节：底座 yaw，绕 z 轴旋转。 -->
      <body name="yaw_link" pos="0 0 0.08">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom name="base_yaw_marker" type="sphere" size="0.045" mass="0.05" material="mat_joint"/>

        <!-- 第 2 个关节：肩 pitch，绕 y 轴旋转。 -->
        <body name="upper_arm" pos="0 0 0.08">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-1.4 1.4"/>
          <geom name="shoulder_marker" type="sphere" size="0.045" mass="0.05" material="mat_joint"/>
          <geom name="upper_arm_geom" type="capsule"
                fromto="0 0 0 0.35 0 0" size="0.035" mass="0.8" material="mat_link1"/>

          <!-- 第 3 个关节：肘 pitch，绕 y 轴旋转。 -->
          <body name="forearm" pos="0.35 0 0">
            <joint name="elbow_pitch" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
            <geom name="elbow_marker" type="sphere" size="0.04" mass="0.05" material="mat_joint"/>
            <geom name="forearm_geom" type="capsule"
                  fromto="0 0 0 0.30 0 0" size="0.028" mass="0.5" material="mat_link2"/>

            <!-- site 是 MuJoCo 里常用的“参考点/末端点”。IK 追的就是这个点。 -->
            <site name="ee_site" pos="0.30 0 0" size="0.035" material="mat_ee"/>
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


class SimpleArmIKDemo:
    """把 FK、Jacobian IK、PD 控制封装在一个小 demo 里。"""

    def __init__(self, target_pos):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        # 目标点存在 mocap body 中，所以可以直接改 data.mocap_pos[0]。
        self.target_pos = np.asarray(target_pos, dtype=float)
        self.data.mocap_pos[0] = self.target_pos

        # 找到末端 site 的 id。后面 data.site_xpos[ee_site_id] 就是末端世界坐标。
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # 找到 3 个关节的 qpos/qvel 下标。
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

        # 当前 IK 算出来的目标关节角。PD 控制器会追它。
        self.target_q = self.data.qpos[self.qpos_ids].copy()

        # 给一个不完全伸直的初始姿态，避免一开始就在奇异姿态附近。
        self.data.qpos[self.qpos_ids] = np.array([0.0, 0.35, -0.65])
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self):
        """读取末端执行器当前世界坐标。"""
        return self.data.site_xpos[self.ee_site_id].copy()

    def solve_ik_one_step(self):
        """
        做一步 Jacobian IK。

        核心公式感：
            error = target_pos - current_ee_pos
            dq = J_pseudo_inverse * error
            target_q = current_q + dq

        这里的 dq 不是最终电机命令，而是“关节角应该往哪里改一点”。
        """
        ee_pos = self.get_ee_pos()
        error = self.target_pos - ee_pos

        # jacp 是 3 x nv 矩阵：
        # 它描述“每个自由度动一点，末端 xyz 会怎么变”。
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # 这个机械臂只有 3 个关节，但 MuJoCo 的 jacp 是对所有 dof 的。
        # 所以取出我们关心的 3 列。
        J = jacp[:, self.dof_ids]

        # damped least squares：带阻尼的伪逆。
        # 阻尼可以避免奇异姿态附近数值爆炸。
        damping = 0.08
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * damping * np.eye(3))

        # error 太大时不要一次改太猛，避免机械臂抖动或撞限位。
        dq = J_pinv @ error
        dq = np.clip(dq, -0.08, 0.08)

        current_q = self.data.qpos[self.qpos_ids].copy()
        self.target_q = current_q + dq

        # 尊重关节范围。真实机器人 IK 也必须考虑 joint limit。
        lower = self.joint_ranges[:, 0]
        upper = self.joint_ranges[:, 1]
        self.target_q = np.clip(self.target_q, lower, upper)

        return error

    def apply_pd_control(self):
        """
        用 PD 控制把关节推向 target_q。

        IK 算的是“希望关节去哪里”。
        PD 控制负责把这个目标变成 motor ctrl，也就是 data.ctrl。
        """
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]

        kp = np.array([35.0, 45.0, 35.0])
        kd = np.array([4.0, 5.0, 4.0])

        torque = kp * (self.target_q - q) - kd * dq
        self.data.ctrl[:] = np.clip(torque, -40.0, 40.0)

    def step(self):
        """一帧完整闭环：IK 算目标关节角 -> PD 写 ctrl -> mj_step 推进。"""
        error = self.solve_ik_one_step()
        self.apply_pd_control()
        mujoco.mj_step(self.model, self.data)
        return error


def parse_args():
    parser = argparse.ArgumentParser(description="Simple MuJoCo Jacobian IK demo.")
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_TARGET_POS,
        help="target end-effector position in world coordinates",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demo = SimpleArmIKDemo(target_pos=np.array(args.target, dtype=float))

    print("Simple arm IK demo")
    print("target_pos:", demo.target_pos)
    print("joint_names:", demo.joint_names)
    print("核心链路: qpos -> FK/site_xpos -> error -> Jacobian IK -> target_q -> PD -> data.ctrl")
    print("-" * 80)

    with viewer.launch_passive(demo.model, demo.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0.32, 0.05, 0.28]
        v.cam.distance = 1.35
        v.cam.azimuth = 135
        v.cam.elevation = -25

        step_count = 0
        while v.is_running():
            error = demo.step()
            step_count += 1

            # 打印慢一点：每 100 step 打一次，也就是大约 0.5 秒一次。
            if step_count % 100 == 0:
                ee_pos = demo.get_ee_pos()
                print("ee_pos:", np.round(ee_pos, 3))
                print("target:", np.round(demo.target_pos, 3))
                print("error_norm:", round(float(np.linalg.norm(error)), 4))
                print("qpos:", np.round(demo.data.qpos[demo.qpos_ids], 3))
                print("target_q:", np.round(demo.target_q, 3))
                print("ctrl:", np.round(demo.data.ctrl, 3))
                print("-" * 80)

            v.sync()
            time.sleep(demo.model.opt.timestep)


if __name__ == "__main__":
    main()

