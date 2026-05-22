"""MADDPG-lite components.

This file provides compact actor/critic networks and a deterministic action
wrapper used by the SE-WSN controller. It is intentionally lightweight so the
repository can be executed on a laptop. A full production MADDPG trainer can
be added without changing the environment/logging API.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, joint_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class DeterministicAction:
    next_hop: np.ndarray
    tx_power_dbm: np.ndarray
    duty_cycle: np.ndarray
    control_messages: int
