# -*- coding: utf-8 -*-
"""
MuJoCo -> ROS2 -> MuJoCo 最小桥接 Demo。

这个 node 做三件事：
    1. 跑 MuJoCo 仿真。
    2. 把 MuJoCo 的 qpos/qvel 转成 ROS2 的 JointState，发布到 /joint_states。
    3. 订阅 /joint_command，把收到的 action 写入 data.ctrl。

运行：
    cd /mnt/c/Users/Administrator/Desktop/robo/03_ros2_mujoco_bridge
    source /opt/ros/humble/setup.bash
    python3 mujoco_node.py
"""

from pathlib import Path

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


ROOT = Path(__file__).resolve().parent
ROBOT_XML = ROOT / "robots" / "mini_humanoid_motor.xml"


class MujocoNode(Node):
    """一个把 MuJoCo 仿真包装成 ROS2 通信节点的对象。"""

    def __init__(self):
        # 注册 ROS2 node 名字。可以用 ros2 node list 看到它。
        super().__init__("mujoco_node")

        # model 是 MuJoCo 静态模型：body、joint、geom、actuator 等结构都在这里。
        # data 是 MuJoCo 运行状态：qpos、qvel、ctrl、xpos、接触信息等每步变化的数据。
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.data = mujoco.MjData(self.model)

        # 整理 actuator -> joint -> qpos/qvel 的映射。
        # 后面发布 JointState 和写 data.ctrl 都要靠这张表对齐顺序。
        self.actuator_table = self._build_actuator_table()
        self.joint_names = [item["joint_name"] for item in self.actuator_table]

        # 找到 torso 的 body id，用来发布 /robot_pose。
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # 发布 /joint_states：
        # 标准机器人关节状态 topic，消息类型是 sensor_msgs/msg/JointState。
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)

        # 发布 /robot_pose：
        # 教学用，表示 torso 在世界坐标系下的位置和姿态。
        self.pose_pub = self.create_publisher(Pose, "/robot_pose", 10)

        # 订阅 /joint_command：
        # policy_node 会发布一个 action 向量，这里收到后写入 data.ctrl。
        self.command_sub = self.create_subscription(
            Float64MultiArray,
            "/joint_command",
            self.on_joint_command,
            10,
        )

        # latest_action 保存最近一次收到的控制命令。
        # 如果 policy_node 还没发命令，就用 0 action。
        self.latest_action = np.zeros(self.model.nu, dtype=float)

        # 100Hz 定时器。每 0.01 秒走一次 step_and_publish。
        # 它就是这个 node 的主循环。
        self.timer = self.create_timer(0.01, self.step_and_publish)

        self.get_logger().info("MuJoCo node started")
        self.get_logger().info(f"publishing /joint_states with {len(self.joint_names)} joints")
        self.get_logger().info(f"listening /joint_command with action size {self.model.nu}")

    def _build_actuator_table(self):
        """建立 motor 和 joint 的对应关系。"""
        table = []

        for actuator_id in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )

            # actuator_trnid[actuator_id, 0] 对 motor 来说就是它控制的 joint id。
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
                }
            )

        return table

    def on_joint_command(self, msg):
        """收到 /joint_command 后，把 ROS2 action 消息缓存起来。"""
        action = np.asarray(msg.data, dtype=float)

        if action.shape != (self.model.nu,):
            self.get_logger().warn(
                f"ignore /joint_command: expected {self.model.nu} values, got {action.shape}"
            )
            return

        # XML 里 motor ctrlrange 是 [-12, 12]，这里再 clip 一次，防止命令过大。
        self.latest_action = np.clip(action, -12.0, 12.0)

    def step_and_publish(self):
        """主循环：写控制 -> MuJoCo 步进 -> 发布观测。"""
        # 1. ROS2 command 进入 MuJoCo actuator 控制入口。
        self.data.ctrl[:] = self.latest_action

        # 2. 推进 MuJoCo 物理仿真一步。
        # qpos/qvel/xpos/contact 等都会在这里更新。
        mujoco.mj_step(self.model, self.data)

        # 3. 把新的 MuJoCo 状态发布给 ROS2 世界。
        self.publish_joint_states()
        self.publish_robot_pose()

    def publish_joint_states(self):
        """把 MuJoCo 的 qpos/qvel 转成 sensor_msgs/msg/JointState。"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        positions = []
        velocities = []
        efforts = []

        for item in self.actuator_table:
            positions.append(float(self.data.qpos[item["qpos_id"]]))
            velocities.append(float(self.data.qvel[item["dof_id"]]))

            # 这里的 effort 用当前 ctrl 表示“电机命令力矩”。
            # 严格物理受力可进一步读 actuator_force，这里先保持教学直观。
            efforts.append(float(self.data.ctrl[item["actuator_id"]]))

        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts

        self.joint_state_pub.publish(msg)

    def publish_robot_pose(self):
        """把 torso 的世界坐标位姿发布成 geometry_msgs/msg/Pose。"""
        pos = self.data.xpos[self.torso_body_id]
        quat = self.data.xquat[self.torso_body_id]

        msg = Pose()
        msg.position.x = float(pos[0])
        msg.position.y = float(pos[1])
        msg.position.z = float(pos[2])

        # MuJoCo xquat 顺序是 [w, x, y, z]。
        # ROS Pose.orientation 顺序是 x, y, z, w。
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        msg.orientation.w = float(quat[0])

        self.pose_pub.publish(msg)


def main():
    rclpy.init()
    node = MujocoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

