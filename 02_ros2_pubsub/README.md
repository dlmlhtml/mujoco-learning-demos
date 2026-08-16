# 02 ROS2 Pub/Sub Demo

This demo is the first ROS2 communication milestone for the MuJoCo learning path.

It contains two plain Python ROS2 nodes:

- `joint_state_publisher_node.py`: publishes virtual robot joint states.
- `joint_state_subscriber_node.py`: subscribes to those joint states and prints them.

The learning goal is to understand:

- A ROS2 node is a running communication-capable program.
- A topic is the named channel used to move data between nodes.
- A message is the structured data format sent on a topic.
- Publisher/subscriber lets two programs communicate without directly calling each other.

## Run

Open Ubuntu/WSL terminal 1:

```bash
cd /mnt/c/Users/Administrator/Desktop/robo/02_ros2_pubsub
source /opt/ros/humble/setup.bash
python3 joint_state_publisher_node.py
```

Open Ubuntu/WSL terminal 2:

```bash
cd /mnt/c/Users/Administrator/Desktop/robo/02_ros2_pubsub
source /opt/ros/humble/setup.bash
python3 joint_state_subscriber_node.py
```

Useful inspection commands:

```bash
ros2 topic list
ros2 topic info /demo/joint_states
ros2 topic echo /demo/joint_states
```

