# 03 ROS2 MuJoCo Bridge

This demo connects MuJoCo simulation data to ROS2 topics and sends a simple command back into MuJoCo.

It is the minimal closed loop:

```text
MuJoCo
  -> read qpos / qvel / torso pose
  -> ROS2 Publisher
  -> /joint_states and /robot_pose
  -> policy_node subscribes
  -> policy_node publishes /joint_command
  -> mujoco_node writes data.ctrl
  -> MuJoCo
```

## Files

- `mujoco_node.py`: runs MuJoCo, publishes robot state, subscribes to command.
- `policy_node.py`: subscribes to joint states and publishes a simple command.
- `robots/mini_humanoid_motor.xml`: standalone MuJoCo robot model for this demo.

## Topics

| Topic | Message Type | Direction | Meaning |
| --- | --- | --- | --- |
| `/joint_states` | `sensor_msgs/msg/JointState` | `mujoco_node -> policy_node` | Joint names, angles, velocities, and effort placeholders |
| `/robot_pose` | `geometry_msgs/msg/Pose` | `mujoco_node -> observers` | Torso world position and quaternion orientation |
| `/joint_command` | `std_msgs/msg/Float64MultiArray` | `policy_node -> mujoco_node` | Motor command vector written into `data.ctrl` |

## Setup In WSL

ROS2 is installed in Ubuntu/WSL. MuJoCo also needs to be installed in that Linux Python:

```bash
python3 -m pip install --user mujoco numpy
```

## Run

Terminal 1:

```bash
cd /mnt/c/Users/Administrator/Desktop/robo/03_ros2_mujoco_bridge
source /opt/ros/humble/setup.bash
python3 mujoco_node.py
```

Terminal 2:

```bash
cd /mnt/c/Users/Administrator/Desktop/robo/03_ros2_mujoco_bridge
source /opt/ros/humble/setup.bash
python3 policy_node.py
```

Inspection commands:

```bash
ros2 topic list
ros2 topic echo /joint_states
ros2 topic echo /robot_pose
ros2 topic echo /joint_command
ros2 topic info /joint_states
```

## Learning Points

- `qpos` and `qvel` are MuJoCo runtime state arrays.
- `JointState` is the standard ROS2 message for robot joint state.
- `Pose` stores world position plus quaternion orientation.
- MuJoCo quaternion order is `[w, x, y, z]`; ROS Pose quaternion order is `x, y, z, w`.
- `/joint_command` is the action channel. In this demo, it is directly copied into `data.ctrl`.
- TF/tf2 is not implemented here, but conceptually it is the system that keeps relationships between frames such as `world`, `torso`, `camera`, and `end_effector`.

