# -*- coding: utf-8 -*-
"""Behavior Cloning MLP 和数据读取公共函数。"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BCPolicy(nn.Module):
    """监督学习策略：state -> predicted expert action。"""

    def __init__(self, state_dim=19, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


def find_episodes(dataset_dir):
    paths = sorted(Path(dataset_dir).glob("episode_*/trajectory.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode data found in: {dataset_dir}")
    return paths


def load_transitions(episode_paths):
    states = []
    actions = []
    for path in episode_paths:
        with np.load(path, allow_pickle=False) as episode:
            states.append(episode["state"].astype(np.float32))
            actions.append(episode["action"].astype(np.float32))
    return np.concatenate(states), np.concatenate(actions)


def save_checkpoint(path, model, state_mean, state_std):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": model.network[0].in_features,
            "action_dim": model.network[-2].out_features,
            "state_mean": torch.as_tensor(state_mean, dtype=torch.float32),
            "state_std": torch.as_tensor(state_std, dtype=torch.float32),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = BCPolicy(checkpoint["state_dim"], checkpoint["action_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["state_mean"].to(device), checkpoint["state_std"].to(device)


def predict_action(model, state_mean, state_std, observation, device="cpu"):
    state = torch.as_tensor(observation, dtype=torch.float32, device=device)
    normalized = (state - state_mean) / state_std
    with torch.no_grad():
        action = model(normalized)
    return action.cpu().numpy()

