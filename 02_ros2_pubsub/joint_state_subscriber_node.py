"""
ROS2 Python 订阅者教学 Demo：订阅并打印虚拟机器人关节状态。

运行方式：
    cd /mnt/c/Users/Administrator/Desktop/robo/02_ros2_pubsub
    source /opt/ros/humble/setup.bash
    python3 joint_state_subscriber_node.py

这个脚本对应 publisher：
    publisher node  ->  /demo/joint_states  ->  subscriber node
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStatePrinter(Node):
    """一个专门订阅 /demo/joint_states 并打印内容的 ROS2 node。"""

    def __init__(self):
        super().__init__("joint_state_printer")

        # 创建订阅者：
        # - JointState：必须和发布者的消息类型一致。
        # - "/demo/joint_states"：必须和发布者的 topic 名字一致。
        # - self.on_joint_state：收到消息后自动调用的回调函数。
        # - 10：订阅队列长度。
        self.subscription = self.create_subscription(
            JointState,
            "/demo/joint_states",
            self.on_joint_state,
            10,
        )

        self.get_logger().info("Listening to /demo/joint_states")

    def on_joint_state(self, msg):
        """每收到一条 JointState 消息，ROS2 就会自动调用这个函数。"""

        # zip 会把关节名字、角度、速度按相同下标配对。
        # 例如 name[0] 对应 position[0] 和 velocity[0]。
        pairs = zip(msg.name, msg.position, msg.velocity)

        readable_items = []
        for name, position, velocity in pairs:
            readable_items.append(f"{name}: pos={position:+.3f} rad, vel={velocity:+.3f} rad/s")

        print(" | ".join(readable_items))


def main():
    rclpy.init()
    node = JointStatePrinter()

    try:
        # spin 会让订阅者一直运行，等待新消息到来。
        # 一旦 /demo/joint_states 有新消息，就会触发 on_joint_state。
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

