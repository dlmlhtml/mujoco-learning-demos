# 04 Simple Arm IK

This demo teaches the minimum robot kinematics loop in MuJoCo:

```text
joint angles qpos
  -> forward kinematics
end-effector position
  -> position error
Jacobian IK
  -> target joint angles
PD control
  -> data.ctrl
  -> mj_step
```

## Files

- `demo_simple_arm_ik.py`: 3-DOF arm IK demo with detailed Chinese comments.

## Run

```powershell
cd 04_simple_arm_ik
python demo_simple_arm_ik.py
```

Set a target position:

```powershell
python demo_simple_arm_ik.py --target 0.55 0.20 0.35
```

## Learning Points

- Joint space: joint angles stored in MuJoCo `data.qpos`.
- Cartesian space: end-effector position stored in MuJoCo `data.site_xpos`.
- FK: MuJoCo updates the end-effector position from current joint angles.
- IK: use target position error and Jacobian to compute a joint update.
- Jacobian: maps joint velocity/change to end-effector velocity/change.
- Controller: use PD torque to move joints toward the IK target angles.

