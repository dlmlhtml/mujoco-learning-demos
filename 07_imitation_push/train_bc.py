# -*- coding: utf-8 -*-
"""用监督学习训练 Behavior Cloning：state -> expert action。"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bc_model import BCPolicy, find_episodes, load_transitions, save_checkpoint


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "bc_push.pt")
    return parser.parse_args()


def mse(model, states, actions):
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(states), actions).item()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 按episode切分，避免同一条轨迹的相邻帧同时出现在训练集和验证集。
    episode_paths = find_episodes(args.dataset)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episode_paths)
    validation_count = max(1, round(len(episode_paths) * 0.2))
    validation_paths = episode_paths[:validation_count]
    training_paths = episode_paths[validation_count:]
    if not training_paths:
        raise ValueError("need at least two episodes for train/validation split")

    train_states_np, train_actions_np = load_transitions(training_paths)
    val_states_np, val_actions_np = load_transitions(validation_paths)

    # 只用训练集统计归一化参数，防止验证集信息泄漏。
    state_mean_np = train_states_np.mean(axis=0)
    state_std_np = np.maximum(train_states_np.std(axis=0), 1e-6)
    train_states_np = (train_states_np - state_mean_np) / state_std_np
    val_states_np = (val_states_np - state_mean_np) / state_std_np

    train_states = torch.from_numpy(train_states_np.astype(np.float32))
    train_actions = torch.from_numpy(train_actions_np.astype(np.float32))
    val_states = torch.from_numpy(val_states_np.astype(np.float32))
    val_actions = torch.from_numpy(val_actions_np.astype(np.float32))

    loader = DataLoader(
        TensorDataset(train_states, train_actions),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = BCPolicy(state_dim=train_states.shape[1], action_dim=train_actions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    print(
        f"train episodes={len(training_paths)}, val episodes={len(validation_paths)}, "
        f"train transitions={len(train_states)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for states, expert_actions in loader:
            predicted_actions = model(states)
            loss = loss_fn(predicted_actions, expert_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_loss = mse(model, train_states, train_actions)
            val_loss = mse(model, val_states, val_actions)
            print(
                f"epoch {epoch:03d}: train_mse={train_loss:.6f}, "
                f"val_mse={val_loss:.6f}"
            )

    save_checkpoint(args.output, model, state_mean_np, state_std_np)
    print(f"saved BC policy to: {args.output}")


if __name__ == "__main__":
    main()

