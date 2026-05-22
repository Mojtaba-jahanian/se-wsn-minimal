from .common import build_action, greedy_next_hop, distance_to_sink

class DRLRoutingPolicy:
    name = "DRL-Routing"
    def reset(self):
        pass

    def act(self, env, risk=None):
        dist = distance_to_sink(env)
        energy_norm = env.energy / env.energy_model.initial_j
        def score(i, j):
            # Learning-inspired local utility, without graph risk context.
            return 0.50 * env.link_quality[i, j] + 0.25 * energy_norm[j] + 0.25 * (1.0 - dist[j]/(env.area*2**0.5))
        return build_action(env, greedy_next_hop(env, score), power_level=4.0, duty_base=0.95, control_messages=10)
