import numpy as np

class RiskLabeler:
    def __init__(self, energy_threshold=0.05, link_quality_min=0.45, cluster_cv_threshold=0.35):
        self.energy_threshold = energy_threshold
        self.link_quality_min = link_quality_min
        self.cluster_cv_threshold = cluster_cv_threshold

    def node_failure_labels(self, energy: np.ndarray, sink_connected: np.ndarray):
        return ((energy <= self.energy_threshold) | (~sink_connected)).astype(float)

    def link_degradation_labels(self, link_quality: np.ndarray):
        return (link_quality <= self.link_quality_min).astype(float)

    def cluster_instability_label(self, energy: np.ndarray, alive: np.ndarray):
        vals = energy[alive]
        if len(vals) < 2:
            return 1.0
        cv = vals.std() / (vals.mean() + 1e-9)
        return float(cv > self.cluster_cv_threshold)
