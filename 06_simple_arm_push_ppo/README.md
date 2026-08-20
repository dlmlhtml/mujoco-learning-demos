# 06 Simple Arm Push PPO

This demo turns the MuJoCo push task into a Gymnasium-style RL environment and trains PPO on it.

The important change from demo 05:

```text
05 hand-written controller:
observation -> update_task_target / IK / PD -> data.ctrl

06 PPO environment:
observation -> neural network policy -> delta xyz -> Jacobian IK -> target_q -> PD -> data.ctrl
```

## Files

- `simple_arm_push_env.py`: Gymnasium environment for the push task.
- `train_ppo.py`: trains a PPO policy with Stable-Baselines3.
- `play_policy.py`: loads a trained PPO model and visualizes it in MuJoCo.
- `evaluate_policy.py`: measures success rate and final cube-to-goal distance.
- `requirements-ppo.txt`: extra packages needed for PPO training.

## Install

```powershell
conda activate mujoco_learn
pip install -r requirements-ppo.txt
```

## Train

```powershell
cd 06_simple_arm_push_ppo
python train_ppo.py
```

The first training stage uses a fixed cube and goal so the complete RL loop can be
verified reliably. After it works, set `randomize_reset=True` to train generalization.

## Play

```powershell
python play_policy.py
```

## Evaluate

```powershell
python evaluate_policy.py
```

## Learning Points

- `observation`: what the policy can see.
- `action`: what the policy controls. Here it is end-effector `delta xyz`, not raw torque.
- `reward`: training signal.
- `episode`: one reset-to-done rollout.
- `policy`: neural network mapping `obs -> action`.
- `value function`: estimates how good the current state is.
- `PPO`: an on-policy Actor-Critic algorithm with stable clipped updates.

This demo is intentionally small. The goal is not a perfect push policy; the goal is to connect RL class concepts to a MuJoCo robot task.
