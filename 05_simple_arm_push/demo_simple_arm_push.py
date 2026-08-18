# -*- coding: utf-8 -*-
"""
简化机械臂推方块教学 Demo。

这个 demo 比 04_simple_arm_ik 多了“接触”和“任务成功”：

    Observation:
        末端位置 ee_pos
        方块位置 cube_pos
        目标位置 goal_pos

    Controller:
        先让末端移动到方块后方
        再沿着目标方向推方块

    Action:
        IK 算 target_q
        PD 算 data.ctrl

    Contact:
        红色末端球 ee_geom 接触蓝色方块 cube_geom

    Task Success:
        cube_pos 靠近 goal_pos

运行：
    cd C:\\Users\\Administrator\\Desktop\\robo\\05_simple_arm_push
    python demo_simple_arm_push.py
"""

import time

import mujoco
import mujoco.viewer as viewer
import numpy as np


CUBE_START_POS = np.array([0.42, -0.18, 0.045], dtype=float)
GOAL_POS = np.array([0.68, 0.18, 0.01], dtype=float)
SUCCESS_RADIUS = 0.16


XML = """
<mujoco model="simple_arm_push">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.02" limited="true"/>
    <geom friction="1.0 0.02 0.001"/>
    <motor ctrllimited="true" ctrlrange="-45 45"/>
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
    <material name="mat_target" rgba="0.10 0.85 0.25 0.75"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="1.4 -2.0 1.0" xyaxes="0.82 0.57 0 -0.23 0.34 0.91"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor"/>

    <!-- IK 当前追踪点。它是绿色小球，只做可视化，不参与接触。 -->
    <body name="ik_target" mocap="true" pos="0.35 0 0.09">
      <geom name="ik_target_geom" type="sphere" size="0.025"
            contype="0" conaffinity="0" material="mat_target"/>
    </body>

    <!-- 可推动方块。freejoint 让它可以在世界中平移和旋转。 -->
    <body name="cube" pos="0.42 -0.18 0.045">
      <freejoint name="cube_freejoint"/>
      <geom name="cube_geom" type="box" size="0.045 0.045 0.045"
            mass="0.12" material="mat_cube"/>
    </body>

    <!-- 绿色目标区域。它不参与碰撞，只用来判断任务成功。 -->
    <geom name="goal_marker" type="cylinder" pos="0.68 0.18 0.01"
          size="0.16 0.01" contype="0" conaffinity="0" material="mat_goal"/>

    <!-- 机械臂固定在世界中。 -->
    <body name="base" pos="0 0 0.08">
      <geom name="base_geom" type="cylinder" size="0.12 0.08" material="mat_base"/>

      <body name="yaw_link" pos="0 0 0.08">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom name="base_yaw_marker" type="sphere" size="0.045" mass="0.05" material="mat_joint"/>

        <body name="upper_arm" pos="0 0 0.08">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom name="shoulder_marker" type="sphere" size="0.045" mass="0.05" material="mat_joint"/>
          <geom name="upper_arm_geom" type="capsule"
                fromto="0 0 0 0.35 0 0" size="0.035" mass="0.8" material="mat_link1"/>

          <body name="forearm" pos="0.35 0 0">
            <joint name="elbow_pitch" type="hinge" axis="0 1 0" range="-1.9 1.9"/>
            <geom name="elbow_marker" type="sphere" size="0.04" mass="0.05" material="mat_joint"/>
            <geom name="forearm_geom" type="capsule"
                  fromto="0 0 0 0.30 0 0" size="0.028" mass="0.5" material="mat_link2"/>

            <!-- site 只负责读末端位置；geom 才真正参与接触。 -->
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


class SimpleArmPushDemo:
    """一个最小 reach + push 任务。"""

    def __init__(self):
        # 把上面的 MJCF XML 编译成 MuJoCo 的静态模型 model。
        # model 里保存：body、joint、geom、site、actuator 的结构和各种索引表。
        self.model = mujoco.MjModel.from_xml_string(XML)

        # 根据 model 创建运行时数据 data。
        # data 里保存：qpos、qvel、ctrl、xpos、contact 等每一帧会变化的数据。
        self.data = mujoco.MjData(self.model)

        # MuJoCo 内部不会用字符串名字直接访问对象。
        # 它会先把名字转换成整数 id，然后用这个 id 去 data/model 的数组里取值。
        #
        # mj_name2id(model, 对象类型, 对象名字) 的意思是：
        #     在这个 model 里，查找某个类型、某个名字的对象，返回它的整数 id。
        #
        # mjOBJ_SITE 表示“我要找的是 site”。
        # "ee_site" 是 XML 里写的：
        #     <site name="ee_site" .../>
        #
        # 找到 ee_site_id 后，后面就可以用：
        #     data.site_xpos[ee_site_id]
        # 读取末端参考点的世界坐标。
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # 找 cube 这个 body 的 id。
        # body 是层级部件/坐标系，cube body 因为有 freejoint，所以会被物理推动。
        # 后面用：
        #     data.xpos[cube_body_id]
        # 读取方块 body 原点的世界坐标。
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

        # 找红色末端球 geom 的 id。
        # geom 是几何体，它可以参与可视化、质量、碰撞。
        # ee_geom 才是真正和方块发生接触的东西，ee_site 只是参考点。
        self.ee_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ee_geom"
        )

        # 找蓝色方块 geom 的 id。
        # contact 里记录的是 geom1/geom2，所以判断“末端是否碰到方块”要用 geom id。
        self.cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )

        # 找绿色目标区域 geom 的 id。
        # 这里 goal_marker 不参与碰撞，只用它的 model.geom_pos 当任务目标位置。
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )

        # 找方块 freejoint 的 id。
        # freejoint 让 cube 可以在世界中自由平移和旋转。
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint"
        )

        # jnt_qposadr[joint_id] 表示：
        #     这个 joint 的位置数据，从 data.qpos 的哪个下标开始。
        #
        # freejoint 的 qpos 有 7 个数：
        #     [x, y, z, qw, qx, qy, qz]
        # 所以 cube_qpos_id 指向这 7 个数的起始位置。
        self.cube_qpos_id = self.model.jnt_qposadr[cube_joint_id]

        # jnt_dofadr[joint_id] 表示：
        #     这个 joint 的速度数据，从 data.qvel 的哪个下标开始。
        #
        # freejoint 的 qvel 有 6 个数：
        #     [vx, vy, vz, wx, wy, wz]
        # 所以 cube_dof_id 指向这 6 个数的起始位置。
        self.cube_dof_id = self.model.jnt_dofadr[cube_joint_id]

        # 这三个是机械臂真正要控制的关节。
        # 注意：cube_freejoint 也是 joint，但它是物体自由运动用的，不是机械臂控制关节。
        self.joint_names = ["base_yaw", "shoulder_pitch", "elbow_pitch"]

        # qpos_ids 保存这 3 个关节角在 data.qpos 里的下标。
        # dof_ids 保存这 3 个关节速度在 data.qvel 里的下标。
        # joint_ranges 保存每个关节允许转动的范围。
        self.qpos_ids = []
        self.dof_ids = []
        self.joint_ranges = []
        for joint_name in self.joint_names:
            # 先用关节名字查 joint_id。
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )

            # hinge joint 的 qpos 只有 1 个数，就是关节角度。
            # 例如 base_yaw 的角度可能存在 data.qpos[0]。
            self.qpos_ids.append(self.model.jnt_qposadr[joint_id])

            # hinge joint 的 qvel 也只有 1 个数，就是关节角速度。
            self.dof_ids.append(self.model.jnt_dofadr[joint_id])

            # XML 里 joint range 的值，例如 range="-1.5 1.5"。
            # IK 算 target_q 后，要用这个范围防止关节转过头。
            self.joint_ranges.append(self.model.jnt_range[joint_id].copy())

        # 转成 numpy array 后，可以一次性索引：
        #     data.qpos[self.qpos_ids]
        # 直接拿到 3 个机械臂关节角。
        self.qpos_ids = np.array(self.qpos_ids, dtype=int)
        self.dof_ids = np.array(self.dof_ids, dtype=int)
        self.joint_ranges = np.array(self.joint_ranges, dtype=float)

        # target_q 是 IK 算出来的“目标关节角”。
        # 它不是电机力矩，真正写入电机的是后面 PD 算出来的 data.ctrl。
        self.target_q = np.zeros(3)

        # control_target_pos 是当前 IK 想让末端去的位置。
        # approach 阶段它在方块后方；push 阶段它在方块前方。
        # 绿色小球 ik_target 会跟着它走，方便你看到控制器现在追哪里。
        self.control_target_pos = np.array([0.35, -0.25, 0.09], dtype=float)

        # phase 表示任务阶段：
        #     approach = 先到方块后方
        #     push     = 开始推方块
        self.phase = "approach"
        self.step_count = 0

        # 初始化仿真状态。
        self.reset()

    def reset(self):
        """重置机械臂、方块和任务阶段。"""
        # mj_resetData 会把 data 恢复到模型默认状态。
        # 包括 qpos、qvel、ctrl、time、contact 缓存等。
        mujoco.mj_resetData(self.model, self.data)

        # 机械臂初始姿态：让肘部朝下，末端更容易接近桌面上的方块。
        # IK 可能有多组解；初始姿态会影响它收敛到哪一组解。
        self.data.qpos[self.qpos_ids] = np.array([-0.35, -0.35, 1.20])
        self.target_q = self.data.qpos[self.qpos_ids].copy()

        # freejoint qpos = [x, y, z, qw, qx, qy, qz]。
        # 这里手动把方块放回初始位置。
        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 3] = CUBE_START_POS

        # freejoint 的姿态用 quaternion 表示。
        # [1, 0, 0, 0] 表示没有旋转。
        self.data.qpos[self.cube_qpos_id + 3 : self.cube_qpos_id + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        # 清空方块的线速度和角速度，避免 reset 后它还带着上一次的速度飞出去。
        self.data.qvel[self.cube_dof_id : self.cube_dof_id + 6] = 0.0

        self.phase = "approach"
        self.step_count = 0

        # 手动改 qpos/qvel 后，调用 mj_forward 刷新派生数据。
        # 例如 data.xpos、data.site_xpos、contact 相关缓存会基于当前 qpos 更新。
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self):
        """末端执行器红球的世界位置。"""
        # ee_site 是贴在 forearm 末端的参考点。
        # data.site_xpos[site_id] 是 MuJoCo 每帧算好的 site 世界坐标。
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_cube_pos(self):
        """蓝色方块 body 的世界位置。"""
        # data.xpos[body_id] 是 body 原点的世界坐标。
        # cube body 的原点就在方块中心，所以这里可当作 cube_pos。
        return self.data.xpos[self.cube_body_id].copy()

    def get_goal_pos(self):
        """绿色目标区域的世界位置。"""
        # goal_marker 是固定 geom，不会在仿真中运动。
        # 固定 geom 的位置存在 model.geom_pos 里，而不是 data.xpos。
        return self.model.geom_pos[self.goal_geom_id].copy()

    def compute_push_direction(self):
        """计算从方块指向目标的水平推送方向。"""
        # 这里只取 xy，因为这是桌面平面上的推方块任务。
        # z 方向由接触高度 contact_z 单独控制。
        cube_xy = self.get_cube_pos()[:2]
        goal_xy = self.get_goal_pos()[:2]
        direction = goal_xy - cube_xy
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([1.0, 0.0])

        # 单位方向向量：长度为 1，只表示方向。
        # 后面乘 0.11 / 0.18，就能得到“方块后方/前方多少距离”的点。
        return direction / norm

    def update_task_target(self):
        """
        根据任务阶段更新 IK 追踪点。

        approach:
            先去方块后方，不要从侧面乱撞。

        push:
            末端继续沿“方块 -> 目标”的方向走，推动方块。
        """
        cube_pos = self.get_cube_pos()
        push_dir = self.compute_push_direction()

        # 方块中心高度约 0.045，末端球半径 0.035。
        # 末端 target z 设到 0.085 左右，更容易碰到方块侧面。
        contact_z = 0.085

        # behind_cube 是“方块后方”的点。
        # 如果目标在方块前方，那么推方块应该先把末端放到方块后面。
        #
        # 公式：
        #     方块中心 - 推送方向 * 后退距离
        #
        # 例子：
        #     方块要往右推，末端就先去方块左边。
        behind_cube = cube_pos.copy()
        behind_cube[:2] -= push_dir * 0.11
        behind_cube[2] = contact_z

        # push_point 是“方块前方”的点。
        # 当末端已经到方块后方后，IK target 会切到这个点。
        # 末端为了追这个点，就会穿过方块附近，于是产生接触并推动方块。
        push_point = cube_pos.copy()
        push_point[:2] += push_dir * 0.18
        push_point[2] = contact_z

        if self.phase == "approach":
            self.control_target_pos = behind_cube

            # 末端足够接近方块后方，就进入 push 阶段。
            # 这个阈值不是物理定律，是任务阶段切换条件。
            if np.linalg.norm(self.get_ee_pos() - behind_cube) < 0.04:
                self.phase = "push"
        else:
            self.control_target_pos = push_point

        # 移动绿色小球，让你看到 IK 现在追的是哪里。
        self.data.mocap_pos[0] = self.control_target_pos

    def solve_ik_one_step(self):
        """用 Jacobian IK 把末端往 control_target_pos 推近一点。"""
        # 当前末端位置来自 FK 的结果：
        #     当前 qpos -> MuJoCo 更新 data.site_xpos -> ee_pos
        ee_pos = self.get_ee_pos()

        # error 是 Cartesian space 里的误差：
        #     希望末端去的位置 - 当前末端位置
        # 它告诉我们末端还差多少 xyz。
        error = self.control_target_pos - ee_pos

        # jacp: position Jacobian，描述关节速度/变化对末端 xyz 的影响。
        # jacr: rotation Jacobian，描述关节速度/变化对末端旋转的影响。
        # 当前 demo 只控制末端位置，所以后面只用 jacp。
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        # 让 MuJoCo 计算 ee_site 这个参考点的 Jacobian。
        # 调用后 jacp/jacr 会被原地填充，不是通过 return 返回。
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # jacp 是对“模型全部自由度”的 Jacobian。
        # 但这里 IK 只控制机械臂 3 个关节，不控制 cube_freejoint。
        # 所以只取 self.dof_ids 对应的 3 列。
        J = jacp[:, self.dof_ids]

        # damped least squares 伪逆：
        #     用来把末端误差 error 反推成关节调整量 dq。
        #
        # 直觉：
        #     J 告诉你“关节动 -> 末端怎么动”
        #     J_pinv 近似反过来告诉你“末端想动 -> 关节怎么动”
        damping = 0.08
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * damping * np.eye(3))

        # dq 是 IK 建议的关节角变化量。
        # 它不是 data.ctrl，也不是力矩。
        dq = J_pinv @ error

        # 限制每一帧最大关节变化，避免目标点突然变动导致机械臂猛甩。
        dq = np.clip(dq, -0.06, 0.06)

        # 当前关节角 + IK 建议变化量 = 新的目标关节角 target_q。
        current_q = self.data.qpos[self.qpos_ids].copy()
        self.target_q = current_q + dq

        # 限制 target_q 不超过 XML 中的 joint range。
        self.target_q = np.clip(
            self.target_q,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )

        return error

    def apply_pd_control(self):
        """把 IK 输出的 target_q 变成电机力矩 data.ctrl。"""
        # 当前关节角 q，来自 MuJoCo 状态 data.qpos。
        q = self.data.qpos[self.qpos_ids]

        # 当前关节角速度 dq，来自 MuJoCo 状态 data.qvel。
        dq = self.data.qvel[self.dof_ids]

        # kp 越大，关节越努力追 target_q。
        # kd 越大，关节速度越容易被刹住，抖动更少。
        kp = np.array([42.0, 54.0, 42.0])
        kd = np.array([4.5, 6.0, 4.5])

        # PD 控制公式：
        #     torque = kp * 位置误差 - kd * 当前速度
        #
        # P 项：target_q - q，离目标越远，推力越大。
        # D 项：-dq，速度越快，反向刹车越强。
        torque = kp * (self.target_q - q) - kd * dq

        # motor actuator 的控制入口是 data.ctrl。
        # 这里的 ctrl 可以理解成当前 3 个电机的力矩命令。
        self.data.ctrl[:] = np.clip(torque, -45.0, 45.0)

    def has_ee_cube_contact(self):
        """检查红色末端球和蓝色方块当前是否接触。"""
        # data.ncon 是当前这一帧的接触数量。
        # 没有接触时，data.ncon = 0。
        for i in range(self.data.ncon):
            # data.contact[i] 是第 i 个接触点。
            # contact.geom1 / contact.geom2 是发生接触的两个 geom id。
            contact = self.data.contact[i]
            geom_pair = {contact.geom1, contact.geom2}

            # 集合 {a, b} 不关心顺序。
            # 因为接触可能记录成 ee-cube，也可能记录成 cube-ee。
            if geom_pair == {self.ee_geom_id, self.cube_geom_id}:
                return True
        return False

    def is_success(self):
        """方块进入绿色目标区域附近就算成功。"""
        cube_xy = self.get_cube_pos()[:2]
        goal_xy = self.get_goal_pos()[:2]
        return bool(np.linalg.norm(cube_xy - goal_xy) < SUCCESS_RADIUS)

    def compute_reward(self):
        """
        一个教学版 reward：
        - 方块离目标越远，惩罚越大。
        - 如果末端接触方块，给一点奖励。
        - 成功进入目标区，给额外奖励。
        """
        cube_to_goal = np.linalg.norm(self.get_cube_pos()[:2] - self.get_goal_pos()[:2])
        ee_to_cube = np.linalg.norm(self.get_ee_pos() - self.get_cube_pos())

        reward = -float(cube_to_goal) - 0.2 * float(ee_to_cube)
        if self.has_ee_cube_contact():
            reward += 0.1
        if self.is_success():
            reward += 1.0
        return reward

    def step(self):
        """一帧完整任务闭环。"""
        self.update_task_target()
        error = self.solve_ik_one_step()
        self.apply_pd_control()
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return error


def main():
    demo = SimpleArmPushDemo()

    print("Simple arm push demo")
    print("核心链路: observation -> IK/PD -> data.ctrl -> contact -> cube motion -> success")
    print("cube_start:", CUBE_START_POS)
    print("goal_pos:", GOAL_POS)
    print("-" * 80)

    with viewer.launch_passive(demo.model, demo.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0.38, 0.00, 0.22]
        v.cam.distance = 1.35
        v.cam.azimuth = 135
        v.cam.elevation = -30

        while v.is_running():
            error = demo.step()

            if demo.step_count % 100 == 0:
                cube_pos = demo.get_cube_pos()
                goal_pos = demo.get_goal_pos()
                dist = np.linalg.norm(cube_pos[:2] - goal_pos[:2])

                print("phase:", demo.phase)
                print("ee_pos:", np.round(demo.get_ee_pos(), 3))
                print("ik_target:", np.round(demo.control_target_pos, 3))
                print("cube_pos:", np.round(cube_pos, 3))
                print("distance_to_goal:", round(float(dist), 4))
                print("contact:", demo.has_ee_cube_contact())
                print("reward:", round(demo.compute_reward(), 4))
                print("success:", demo.is_success())
                print("ctrl:", np.round(demo.data.ctrl, 3))
                print("-" * 80)

            if demo.is_success():
                print("Task success. Reset after a short pause.")
                time.sleep(1.0)
                demo.reset()

            v.sync()
            time.sleep(demo.model.opt.timestep)


if __name__ == "__main__":
    main()
