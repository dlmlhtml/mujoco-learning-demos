# 05 Simple Arm Push

This demo extends the IK arm into a small robot manipulation task:

```text
Observation
  -> IK controller
  -> Action/data.ctrl
  -> Contact
  -> Cube motion
  -> Task success
```

## Files

- `demo_simple_arm_push.py`: 3-DOF arm pushes a cube toward a goal area.

## Run

```powershell
cd 05_simple_arm_push
python demo_simple_arm_push.py
```

## Learning Points

- A movable object needs a body with `freejoint`.
- Contact happens between collision geoms, not between sites.
- The red end-effector geom can push the blue cube.
- `data.ncon` and `data.contact` describe current contacts.
- Task success can be defined by object distance to goal.
- Reward often combines distance-to-object, distance-to-goal, and success bonus.

