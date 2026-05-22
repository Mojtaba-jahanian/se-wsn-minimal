import numpy as np
from .common import build_action, greedy_next_hop, distance_to_sink
from sewsn.models.maddpg import DeterministicAction

class SEWSNPolicy:
    name = "SE-WSN"
    def __init__(self, tau_f=0.70, tau_link=0.65, risk_broadcast_interval=10):
        self.tau_f = tau_f
        self.tau_link = tau_link
        self.risk_broadcast_interval = risk_broadcast_interval

    def reset(self):
        pass

    def act(self, env, risk=None):
        if risk is None:
            node_risk = np.zeros(env.n)
            link_risk = np.zeros((env.n, env.n))
        else:
            node_risk = risk["node_risk"]
            link_risk = risk["link_risk"]
        energy_norm = env.energy / env.energy_model.initial_j
        dist = distance_to_sink(env)
        def score(i, j):
            return (
                0.42 * env.link_quality[i, j]
                + 0.14 * energy_norm[j]
                + 0.16 * (1.0 - node_risk[j])
                + 0.08 * (1.0 - link_risk[i, j])
                + 0.20 * (1.0 - dist[j] / (env.area * 2**0.5))
            )
        next_hop = greedy_next_hop(env, score)
        # Moderate power increase only on risky outgoing links.
        power = np.ones(env.n) * 4.0
        for i in np.flatnonzero(env.alive):
            j = next_hop[i]
            if j >= 0 and link_risk[i, j] > self.tau_link:
                power[i] = 5.0
        # Duty-cycle regulation for risky low-energy nodes.
        duty = np.ones(env.n) * 0.98
        duty = np.clip(duty - 0.10 * node_risk + 0.04 * energy_norm, 0.70, 1.0)
        duty[~env.alive] = 0.0
        control = 5
        if env.round_id % self.risk_broadcast_interval == 0:
            control += int(env.alive.sum() * 0.25)
        return DeterministicAction(next_hop=next_hop, tx_power_dbm=power, duty_cycle=duty, control_messages=control)
