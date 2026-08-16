"""
ROS2 Python 发布者教学 Demo：发布虚拟机器人关节状态。

运行方式：
    cd /mnt/c/Users/Administrator/Desktop/robo/02_ros2_pubsub
    source /opt/ros/humble/setup.bash
    python3 joint_state_publisher_node.py

你要观察的重点：
    1. 这个文件运行起来以后，就是一个 ROS2 node。
    2. node 里面创建 publisher，把消息发到 /demo/joint_states。
    3. JointState 是消息格式，里面有 name / position / velocity / effort。
    4. timer 每隔固定时间调用一次 publish_joint_state，相当于循环发布。
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class VirtualJointStatePublisher(Node):
    """一个专门发布虚拟关节状态的 ROS2 node。"""

    def __init__(self):
        # super().__init__ 会把当前对象注册成 ROS2 node。
        # "virtual_joint_state_publisher" 是 node 名字，可用 ros2 node list 看到。
        super().__init__("virtual_joint_state_publisher")

        # 创建发布者：
        # - JointState：这个 topic 使用的消息类型。
        # - "/demo/joint_states"：topic 名字，也就是数据频道名。
        # - 10：队列长度。订阅者来不及接收时，ROS2 最多缓存 10 条。
        self.publisher = self.create_publisher(JointState, "/demo/joint_states", 10)

        # 每 0.1 秒调用一次 publish_joint_state，也就是 10Hz 发布频率。
        self.timer = self.create_timer(0.1, self.publish_joint_state)

        # 保存一个仿真时间变量。这里不用真实机器人，只用 sin/cos 造几个会变化的角度。
        self.t = 0.0

        # 这几个名字先模拟一个简化机器人：左右髋、左右膝、左右肩。
        self.joint_names = [
            "left_hip_y",
            "left_knee_y",
            "right_hip_y",
            "right_knee_y",
            "left_shoulder_y",
            "right_shoulder_y",
        ]

        self.get_logger().info("Publishing virtual joint states on /demo/joint_states")

    def publish_joint_state(self):
        """构造一条 JointState 消息，并发布出去。"""

        msg = JointState()

        # header.stamp 是消息时间戳。
        # 后面和 MuJoCo 联动时，它表示这条观测来自哪个时刻。
        msg.header.stamp = self.get_clock().now().to_msg()

        # name：关节名字列表。
        # position / velocity / effort 的每个元素，都按 name 的顺序一一对应。
        msg.name = self.joint_names

        # position：关节角度，单位是 rad（弧度）。
        # 这里让左右腿反向摆动，左右肩也反向摆动，方便你看出数据在变化。
        msg.position = [
            0.35 * math.sin(self.t),
            0.70 + 0.25 * math.sin(self.t),
            -0.35 * math.sin(self.t),
            0.70 - 0.25 * math.sin(self.t),
            0.50 * math.sin(self.t),
            -0.50 * math.sin(self.t),
        ]

        # velocity：关节角速度，单位是 rad/s。
        # 这里是 position 对时间的变化趋势，教学版用 cos 构造。
        msg.velocity = [
            0.35 * math.cos(self.t),
            0.25 * math.cos(self.t),
            -0.35 * math.cos(self.t),
            -0.25 * math.cos(self.t),
            0.50 * math.cos(self.t),
            -0.50 * math.cos(self.t),
        ]

        # effort：关节力矩/力，单位通常是 N*m。
        # 这里还没有真实电机控制，所以先填 0，表示只是发布状态观测。
        msg.effort = [0.0 for _ in self.joint_names]

        self.publisher.publish(msg)

        # 每次发布后让时间前进一点。这个变量类似我们 MuJoCo demo 里的仿真时间。
        self.t += 0.1


def main():
    # 初始化 ROS2 Python 客户端。
    rclpy.init()

    # 创建 node 对象。到这里 node 还只是创建好了，没有开始循环。
    node = VirtualJointStatePublisher()

    try:
        # spin 会让 node 一直运行，等待 timer/callback 被触发。
        # 对这个发布者来说，spin 期间 timer 会不断调用 publish_joint_state。
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 停止程序前销毁 node，并关闭 rclpy。
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

