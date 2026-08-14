# 01 RL/VLA Env

This demo is the first curated milestone: a MuJoCo environment scaffold shaped like a future RL/VLA training interface.

## What It Teaches

- How to wrap a MuJoCo simulation into an environment class
- How `reset()` initializes an episode
- How `step(action)` applies motor commands and advances physics
- How to return `obs`, `reward`, `terminated`, `truncated`, and `info`
- How to expose both low-dimensional state and rendered image observations

## Run

```powershell
python demo_scene_env_v2.py
```

## Files

- `demo_scene_env_v2.py`: environment wrapper and viewer demo
- `robots/mini_humanoid_motor.xml`: MJCF robot model with motors
