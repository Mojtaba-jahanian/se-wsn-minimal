from dataclasses import dataclass
import numpy as np

@dataclass
class LinkModel:
    base_range_m: float = 85.0
    path_loss_exp: float = 2.2
    lq_noise_std: float = 0.035
    min_lq: float = 0.05

    def distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        diff = positions[:, None, :] - positions[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    def link_quality(self, distances: np.ndarray, rng: np.random.Generator, interference: float = 0.0) -> np.ndarray:
        # Smooth link-quality model in [0, 1].
        normalized = distances / max(self.base_range_m, 1e-6)
        lq = np.exp(-(normalized ** self.path_loss_exp))
        lq += rng.normal(0.0, self.lq_noise_std, size=distances.shape)
        lq -= interference
        lq = np.clip(lq, self.min_lq, 1.0)
        np.fill_diagonal(lq, 0.0)
        return lq

    def adjacency(self, lq: np.ndarray, threshold: float = 0.35) -> np.ndarray:
        adj = (lq >= threshold).astype(float)
        np.fill_diagonal(adj, 0.0)
        return adj
