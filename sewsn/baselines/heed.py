import numpy as np
from .common import build_action, greedy_next_hop, distance_to_sink

class HEEDPolicy:
    name = "HEED"
    def __init__(self, recluster_interval=200):
        self.recluster_interval = recluster_interval

    def reset(self):
        pass

    def act(self, env, risk=None):
        dist = distance_to_sink(env)
        energy_norm = env.energy / env.energy_model.initial_j
        control = int(env.alive.sum() * 0.55) if env.round_id % self.recluster_interval == 0 else 7
        def score(i, j):
            return 0.42 * energy_norm[j] + 0.38 * env.link_quality[i, j] + 0.20 * (1.0 - dist[j]/(env.area*np.sqrt(2)))
        return build_action(env, greedy_next_hop(env, score), power_level=3.0, control_messages=control)
