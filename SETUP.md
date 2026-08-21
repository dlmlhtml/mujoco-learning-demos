# Reproducible Setup

This repository has two environment groups:

- Native Python demos: `01`, `04`, `05`, `06`, and `07`.
- ROS2 demos: `02` and `03`, which require Ubuntu/WSL and ROS2 Humble.

The Conda environment below covers the native Python demos. It does not install ROS2.

## 1. Clone

```powershell
git clone https://github.com/dlmlhtml/mujoco-learning-demos.git
cd mujoco-learning-demos
```

## 2. Create The Python Environment

Recommended one-command Conda setup:

```powershell
conda env create -f environment.yml
conda activate mujoco_learn
python verify_setup.py
```

To update an existing environment from the same file:

```powershell
conda env update -f environment.yml --prune
conda activate mujoco_learn
python verify_setup.py
```

Alternative setup with an existing Python 3.10 environment:

```powershell
python -m pip install -r requirements.txt
python verify_setup.py
```

The verified dependency set is:

| Dependency | Version |
| --- | --- |
| Python | 3.10.20 |
| MuJoCo | 3.11.0 |
| NumPy | 2.2.6 |
| Gymnasium | 1.3.0 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.13.0 CPU build |
| TensorBoard | 2.21.0 |

Exact Python patch and PyTorch build tags may differ by operating system. The package
versions in `requirements.txt` and `environment.yml` match the environment used to
verify the demos.

## 3. Run The Native Demos

Run commands from each numbered folder:

```powershell
cd 01_rl_vla_env
python demo_scene_env_v2.py
```

The same environment runs `04_simple_arm_ik` and `05_simple_arm_push` directly.

Demo `06` does not include its generated PPO checkpoint. Recreate it with:

```powershell
cd 06_simple_arm_push_ppo
python train_ppo.py
python evaluate_policy.py
python play_policy.py
```

Demo `07` does not include demonstration data or its generated BC checkpoint.
Recreate both with:

```powershell
cd 07_imitation_push
python collect_expert_data.py --episodes 50
python train_bc.py
python evaluate_bc.py
python play_bc_policy.py
```

The viewer scripts require a desktop session with working OpenGL graphics. Training
and non-visual evaluation can run without opening the interactive viewer.

## 4. ROS2 Demos

Python requirements do not install ROS2. Use Ubuntu 22.04, either natively or through
WSL2, and install ROS2 Humble separately.

In every new Ubuntu terminal:

```bash
source /opt/ros/humble/setup.bash
```

For the MuJoCo bridge, install its Python dependencies inside the Ubuntu Python,
not inside the Windows Conda environment:

```bash
cd /mnt/c/path/to/mujoco-learning-demos/03_ros2_mujoco_bridge
python3 -m pip install --user -r requirements-wsl.txt
```

Then follow the README files in `02_ros2_pubsub` and `03_ros2_mujoco_bridge`.

## 5. Generated Files

The following are intentionally excluded from Git:

```text
dataset/
models/
runs/
__pycache__/
```

This keeps the repository small. A fresh clone contains the code needed to regenerate
those artifacts, but it does not contain pretrained policies or collected images.

## Troubleshooting

- Use `python -m pip` to ensure packages install into the active Python environment.
- Run `python verify_setup.py` before debugging an individual demo.
- If a viewer fails on a headless machine, use the non-visual evaluation script.
- Keep ROS2's Ubuntu Python separate from the Windows Conda environment.
