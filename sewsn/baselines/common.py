import numpy as np
from sewsn.models.maddpg import DeterministicAction

def candidate_neighbors(env, i):
    return np.flatnonzero((env.adjacency[i] > 0) & env.alive)

def distance_to_sink(env):
    return np.sqrt(((env.positions - env.sink[None, :]) ** 2).sum(axis=1))

def build_action(env, next_hop, power_level=3.0, duty_base=1.0, control_messages=0):
    tx_power = np.ones(env.n) * power_level
    duty = np.ones(env.n) * duty_base
    duty[~env.alive] = 0.0
    return DeterministicAction(next_hop=np.asarray(next_hop, dtype=int), tx_power_dbm=tx_power, duty_cycle=duty, control_messages=control_messages)

def greedy_next_hop(env, score_fn):
    next_hop = np.full(env.n, -1, dtype=int)
    dist_sink = distance_to_sink(env)
    for i in np.flatnonzero(env.alive):
        neigh = candidate_neighbors(env, i)
        if len(neigh) == 0:
            continue
        # Prefer neighbors closer to sink to reduce loops.
        neigh = neigh[dist_sink[neigh] < dist_sink[i] + 10.0]
        if len(neigh) == 0:
            neigh = candidate_neighbors(env, i)
        scores = np.array([score_fn(i, j) for j in neigh])
        next_hop[i] = int(neigh[np.argmax(scores)])
    return next_hop
