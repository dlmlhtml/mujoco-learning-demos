# MuJoCo Learning Demos

Curated MuJoCo learning demos. Each numbered folder is intended to be a self-contained milestone that can be studied and reproduced independently.

## Current Demo

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

## Setup

```powershell
pip install -r requirements.txt
```

Large third-party model downloads and local experiment artifacts are intentionally excluded.
