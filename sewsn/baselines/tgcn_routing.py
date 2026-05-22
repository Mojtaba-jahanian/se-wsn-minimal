from .common import build_action, greedy_next_hop, distance_to_sink

class TGCNRoutingPolicy:
    name = "T-GCN"
    def reset(self):
        pass

    def act(self, env, risk=None):
        if risk is None:
            node_risk = 0.0
            link_risk = env.link_quality * 0.0
        else:
            node_risk = risk["node_risk"]
            link_risk = risk["link_risk"]
        dist = distance_to_sink(env)
        def score(i, j):
            # Temporal risk prediction is used through a heuristic route score.
            return 0.45 * env.link_quality[i, j] + 0.25 * (1.0 - node_risk[j]) + 0.15 * (1.0 - link_risk[i, j]) + 0.15 * (1.0 - dist[j]/(env.area*2**0.5))
        return build_action(env, greedy_next_hop(env, score), power_level=3.0, duty_base=0.9, control_messages=8)
