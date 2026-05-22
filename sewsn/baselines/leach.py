import numpy as np
from .common import build_action, greedy_next_hop, distance_to_sink

class LEACHPolicy:
    name = "LEACH"
    def __init__(self, ch_prob=0.05, recluster_interval=200):
        self.ch_prob = ch_prob
        self.recluster_interval = recluster_interval
        self.cluster_heads = None

    def reset(self):
        self.cluster_heads = None

    def act(self, env, risk=None):
        if self.cluster_heads is None or env.round_id % self.recluster_interval == 0:
            alive = np.flatnonzero(env.alive)
            k = max(1, int(len(alive) * self.ch_prob))
            self.cluster_heads = np.zeros(env.n, dtype=bool)
            if len(alive):
                chosen = env.rng.choice(alive, size=min(k, len(alive)), replace=False)
                self.cluster_heads[chosen] = True
            control = int(env.alive.sum() * 0.8)
        else:
            control = 5
        dist = distance_to_sink(env)
        def score(i, j):
            ch_bonus = 0.35 if self.cluster_heads is not None and self.cluster_heads[j] else 0.0
            return 0.55 * env.link_quality[i, j] + 0.25 * (1.0 - dist[j]/(env.area*np.sqrt(2))) + ch_bonus
        return build_action(env, greedy_next_hop(env, score), power_level=3.0, control_messages=control)
