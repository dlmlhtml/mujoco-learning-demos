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

### `04_simple_arm_ik`

Minimal Jacobian IK demo for a 3-DOF MuJoCo arm.

It includes:

- A target point in Cartesian space
- End-effector position tracking through `site_xpos`
- Jacobian-based IK updates
- PD control that writes motor commands into `data.ctrl`

Run:

```powershell
cd 04_simple_arm_ik
python demo_simple_arm_ik.py
```

### `05_simple_arm_push`

Minimal contact manipulation demo where a simple arm pushes a cube toward a goal.

It includes:

- A movable cube with `freejoint`
- Contact detection through `data.ncon` and `data.contact`
- Two-stage approach/push target logic
- Reward and success calculation based on cube-goal distance

Run:

```powershell
cd 05_simple_arm_push
python demo_simple_arm_push.py
```

### `06_simple_arm_push_ppo`

Gymnasium-style MuJoCo push environment with PPO training scripts.

It includes:

- `SimpleArmPushEnv` with `reset()`, `step(action)`, `render()`, reward, and success logic
- Cartesian `delta xyz` action mapped through Jacobian IK and PD into MuJoCo `data.ctrl`
- 24D observation with joint state, task geometry, push direction, and contact state
- Stable-Baselines3 PPO training and playback scripts

Run:

```powershell
cd 06_simple_arm_push_ppo
pip install -r requirements-ppo.txt
python train_ppo.py
```

### `07_imitation_push`

Random-goal robot demonstrations and state-based Behavior Cloning.

It includes:

- A scripted expert verified on random target directions
- Episode datasets with aligned state, RGB image, action, and next state
- Episode-level train/validation splitting
- A PyTorch MLP trained with supervised action regression
- Closed-loop evaluation and interactive MuJoCo playback

Run:

```powershell
cd 07_imitation_push
python collect_expert_data.py --episodes 50
python train_bc.py
python evaluate_bc.py
python play_bc_policy.py
```

## Setup

Recommended reproducible setup for the native Python demos:

```powershell
conda env create -f environment.yml
conda activate mujoco_learn
python verify_setup.py
```

See [SETUP.md](SETUP.md) for the pip alternative, tested versions, generated model/data
instructions, GUI notes, and the separate ROS2 Humble setup.

Large third-party model downloads and local experiment artifacts are intentionally excluded.
