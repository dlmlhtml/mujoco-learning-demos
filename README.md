# MuJoCo Learning Demos

Curated MuJoCo learning demos. Each numbered folder is intended to be a self-contained milestone that can be studied and reproduced independently.

## Demos

### `01_rl_vla_env`

MuJoCo simulation environment scaffold for RL/VLA-style training.

It includes:

- A fixed-base mini humanoid with motor actuators
- Floor, lights, obstacles, movable cube, and goal marker
- `reset()` returning `(obs, info)`
- `step(action)` returning `(obs, reward, terminated, truncated, info)`
- `render()` returning an RGB image
- Dict observation with `state` and `image`

Run:

```powershell
cd 01_rl_vla_env
python demo_scene_env_v2.py
```

### `02_ros2_pubsub`

Minimal ROS2 Python publisher/subscriber demo using virtual robot joint states.

It includes:

- A publisher node that sends `sensor_msgs/msg/JointState`
- A subscriber node that receives and prints joint names, positions, and velocities
- Useful `ros2 topic` inspection commands

Run in Ubuntu/WSL:

```bash
cd /mnt/c/Users/Administrator/Desktop/robo/02_ros2_pubsub
source /opt/ros/humble/setup.bash
python3 joint_state_publisher_node.py
```

### `03_ros2_mujoco_bridge`

Minimal MuJoCo + ROS2 bridge loop.

It includes:

- `mujoco_node.py` publishing `/joint_states` and `/robot_pose`
- `policy_node.py` subscribing to `/joint_states`
- `/joint_command` feedback into MuJoCo `data.ctrl`
- A standalone mini humanoid MJCF model

Run in Ubuntu/WSL:

```bash
cd /mnt/c/Users/Administrator/Desktop/robo/03_ros2_mujoco_bridge
source /opt/ros/humble/setup.bash
python3 mujoco_node.py
```

## Setup

```powershell
pip install -r requirements.txt
```

Large third-party model downloads and local experiment artifacts are intentionally excluded.
