# 07 Imitation Push

This demo introduces robot demonstration data and Behavior Cloning (BC).

```text
random task state
        ↓
scripted expert
        ↓
state_t / image_t / expert_action_t dataset
        ↓
supervised learning
        ↓
BC policy: state → predicted action
```

## Files

- `imitation_push_env.py`: random-goal MuJoCo task and Gymnasium interface.
- `scripted_expert.py`: rule-based expert that generates action labels.
- `collect_expert_data.py`: saves one directory per demonstration episode.
- `bc_model.py`: MLP policy, dataset loading, checkpoint helpers.
- `train_bc.py`: supervised state-to-action training.
- `evaluate_bc.py`: closed-loop evaluation on unseen random seeds.
- `play_bc_policy.py`: interactive MuJoCo playback.

## Dataset

Each episode contains:

```text
dataset/episode_0000/
├── trajectory.npz
└── metadata.json
```

`trajectory.npz` contains aligned arrays:

- `state[t]`: robot/task state before action `t`.
- `image[t]`: RGB observation before action `t`.
- `action[t]`: expert action computed from `state[t]`.
- `next_state[t]`: state after executing `action[t]`.
- `reward`, `terminated`, `truncated`: transition results.

The first BC model trains only on `state → action`. Images and instruction text are
saved now so the same data structure can later be extended toward visual policies.

## Run

```powershell
conda activate mujoco_learn
cd C:\Users\Administrator\Desktop\robo\07_imitation_push
python collect_expert_data.py --episodes 50
python train_bc.py
python evaluate_bc.py
python play_bc_policy.py
```

## Learning points

- A scripted expert has no learned parameters; it creates supervised action labels.
- Behavior Cloning minimizes prediction error against expert actions.
- Train/validation splitting is done by episode, not by adjacent frames.
- Low validation MSE does not guarantee closed-loop task success.
- Closed-loop errors can compound because BC may visit states absent from the dataset.

