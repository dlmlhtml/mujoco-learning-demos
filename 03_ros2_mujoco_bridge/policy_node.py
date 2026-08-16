# -*- coding: utf-8 -*-
"""
ROS2 policy_node 教学 Demo。

这个 node 做两件事：
    1. 订阅 /joint_states，读取 MuJoCo 发布的关节角度和速度。
    2. 发布 /joint_command，把一个简单控制向量发回 mujoco_node。

这里的 policy 还不是神经网络，只是一个最小可观察闭环：
    observation -> simple policy -> action

运行：
    cd /mnt/c/Users/Administrator/Desktop/robo/03_ros2_mujoco_bridge
    source /opt/ros/humble/setup.bash
    python3 policy_node.py
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class PolicyNode(Node):
    """一个订阅机器人状态并发布动作命令的 ROS2 node。"""

    def __init__(self):
        super().__init__("policy_node")

        # 订阅 MuJoCo 发出来的关节状态。
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_states,
            10,
        )

        # 发布动作命令给 MuJoCo。
        # 这里用 Float64MultiArray 表示 action vector，长度要和 MuJoCo motor 数一致。
        self.command_pub = self.create_publisher(Float64MultiArray, "/joint_command", 10)

        self.last_print_time = 0.0
        self.get_logger().info("policy_node started")
        self.get_logger().info("listening /joint_states and publishing /joint_command")

    def on_joint_states(self, msg):
        """收到 JointState 后，读取 observation，并生成 action。"""
        now = self.get_clock().now().nanoseconds * 1e-9

        # msg.name / msg.position / msg.velocity 是同下标对应的。
        # 这里先转成 dict，方便按关节名字取值。
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity))

        # 每 1 秒打印一次，避免刷屏太快。
        if now - self.last_print_time > 1.0:
            self.last_print_time = now
            left_knee = positions.get("left_knee_pitch", 0.0)
            right_knee = positions.get("right_knee_pitch", 0.0)
            self.get_logger().info(
                f"obs: left_knee={left_knee:+.3f} rad, right_knee={right_knee:+.3f} rad"
            )

        # 生成一个简单 action。
        # 顺序必须和 mujoco_node 的 actuator_table 一致：
        # [left_shoulder, right_shoulder, left_hip, left_knee, left_ankle,
        #  right_hip, right_knee, right_ankle]
        action = self.simple_policy(now, positions, velocities)

        cmd = Float64MultiArray()
        cmd.data = action
        self.command_pub.publish(cmd)

    def simple_policy(self, t, positions, velocities):
        """一个手写策略：让肩膀和膝盖做轻微周期动作。"""
        wave = math.sin(t * 1.5)

        # 这里为了教学清楚，直接发布 motor command。
        # 对当前 MJCF 的 motor 来说，/joint_command 最终会进入 data.ctrl。
        action = [0.0] * 8
        action[0] = -1.5 * wave   # left_shoulder_pitch_motor
        action[1] = 1.5 * wave    # right_shoulder_pitch_motor
        action[2] = 0.8 * wave    # left_hip_pitch_motor
        action[3] = 1.2 * wave    # left_knee_pitch_motor
        action[4] = -0.5 * wave   # left_ankle_pitch_motor
        action[5] = -0.8 * wave   # right_hip_pitch_motor
        action[6] = -1.2 * wave   # right_knee_pitch_motor
        action[7] = 0.5 * wave    # right_ankle_pitch_motor

        return action


def main():
    rclpy.init()
    node = PolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

