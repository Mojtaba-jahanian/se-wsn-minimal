from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass
class GraphSnapshot:
    round_id: int
    features: np.ndarray
    adjacency: np.ndarray
    link_quality: np.ndarray

class TemporalGraphBuilder:
    def __init__(self, window: int = 4):
        self.window = int(window)
        self.snapshots = deque(maxlen=self.window)

    def add(self, round_id: int, features: np.ndarray, adjacency: np.ndarray, link_quality: np.ndarray):
        snap = GraphSnapshot(round_id, features.copy(), adjacency.copy(), link_quality.copy())
        self.snapshots.append(snap)
        return snap

    def sequence(self):
        return list(self.snapshots)

    def stacked_features(self):
        if not self.snapshots:
            return None
        return np.stack([s.features for s in self.snapshots], axis=0)

    def latest(self):
        if not self.snapshots:
            return None
        return self.snapshots[-1]
