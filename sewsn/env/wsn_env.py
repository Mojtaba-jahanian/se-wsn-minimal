from dataclasses import dataclass
import heapq
from typing import Dict, Any
import numpy as np
from .energy_model import EnergyModel
from .link_model import LinkModel

@dataclass
class EnvState:
    round_id: int
    positions: np.ndarray
    energy: np.ndarray
    alive: np.ndarray
    link_quality: np.ndarray
    adjacency: np.ndarray
    traffic: np.ndarray

class WSNEnv:
    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        exp = config["experiment"]
        self.n = int(exp.get("n_nodes", 100))
        self.area = float(exp.get("area_m", 300))
        self.sink = np.array(exp.get("sink", [self.area/2, self.area/2]), dtype=float)
        self.packet_rate = float(exp.get("packet_rate", 1.0))
        self.max_hops = int(exp.get("max_hops", 8))
        self.round_id = 0
        self.energy_model = EnergyModel(**config.get("energy", {}))
        radio_cfg = config.get("radio", {})
        self.power_levels = np.array(radio_cfg.pop("tx_power_levels_dbm", [0, 1, 3, 4, 5]), dtype=float)
        self.link_model = LinkModel(**radio_cfg)
        self.positions = self.rng.uniform(0, self.area, size=(self.n, 2))
        self._dist_matrix = self.link_model.distance_matrix(self.positions)
        self._sink_dist = np.sqrt(((self.positions - self.sink[None, :]) ** 2).sum(axis=1))
        self.energy = np.full(self.n, self.energy_model.initial_j, dtype=float)
        self.alive = np.ones(self.n, dtype=bool)
        self.traffic = np.ones(self.n, dtype=float) * self.packet_rate
        self.link_quality = np.zeros((self.n, self.n), dtype=float)
        self.adjacency = np.zeros((self.n, self.n), dtype=float)
        self.refresh_links()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.round_id = 0
        self.positions = self.rng.uniform(0, self.area, size=(self.n, 2))
        self._dist_matrix = self.link_model.distance_matrix(self.positions)
        self._sink_dist = np.sqrt(((self.positions - self.sink[None, :]) ** 2).sum(axis=1))
        self.energy = np.full(self.n, self.energy_model.initial_j, dtype=float)
        self.alive = np.ones(self.n, dtype=bool)
        self.traffic = np.ones(self.n, dtype=float) * self.packet_rate
        self.refresh_links()
        return self.state()

    def refresh_links(self):
        interference = 0.015 * np.sin(self.round_id / 180.0)
        self.link_quality = self.link_model.link_quality(self._dist_matrix, self.rng, interference=max(0, interference))
        self.adjacency = self.link_model.adjacency(self.link_quality, threshold=0.35)
        dead = ~self.alive
        self.adjacency[dead, :] = 0
        self.adjacency[:, dead] = 0
        self.link_quality[dead, :] = 0
        self.link_quality[:, dead] = 0

    def state(self) -> EnvState:
        return EnvState(self.round_id, self.positions.copy(), self.energy.copy(), self.alive.copy(), self.link_quality.copy(), self.adjacency.copy(), self.traffic.copy())

    def sink_quality(self) -> np.ndarray:
        lq = np.exp(-((self._sink_dist / self.link_model.base_range_m) ** self.link_model.path_loss_exp))
        return np.clip(lq, 0.0, 1.0)

    def features(self) -> np.ndarray:
        degree = self.adjacency.sum(axis=1)
        denom = np.maximum(degree, 1.0)
        mean_lq = self.link_quality.sum(axis=1) / denom
        energy_norm = self.energy / self.energy_model.initial_j
        return np.stack([energy_norm, mean_lq, degree / max(self.n - 1, 1), self.traffic, self.sink_quality()], axis=1)

    def sink_connected_mask(self) -> np.ndarray:
        # A node is sink-connected if it can directly reach sink or a live multi-hop path to a direct node.
        sink_lq = self.sink_quality()
        direct = (sink_lq >= 0.35) & self.alive
        connected = direct.copy()
        changed = True
        while changed:
            prev = connected.copy()
            reachable = (self.adjacency @ connected.astype(float)) > 0
            connected = connected | (reachable & self.alive)
            changed = not np.array_equal(prev, connected)
        return connected

    def step(self, actions):
        self.round_id += 1
        next_hop = actions.next_hop
        tx_power = actions.tx_power_dbm
        duty_cycle = actions.duty_cycle
        generated = int(self.alive.sum() * self.packet_rate)
        delivered = 0
        control_packets = int(actions.control_messages)
        retransmissions = 0
        for src in np.flatnonzero(self.alive):
            path = self._follow_path(src, next_hop)
            if path is None:
                continue
            ok = True
            for u, v in zip(path[:-1], path[1:]):
                # v == -1 means sink.
                if v == -1:
                    lq = self.sink_quality()[u]
                else:
                    lq = self.link_quality[u, v]
                p_success = np.clip(0.55 + 0.45 * lq + 0.025 * tx_power[u], 0.05, 0.98)
                if self.rng.random() > p_success:
                    ok = False
                    retransmissions += 1
                self.energy[u] -= self.energy_model.tx_cost(tx_power[u], lq)
                if v >= 0:
                    self.energy[v] -= self.energy_model.rx_cost()
            if ok:
                delivered += 1
        # idle and control cost
        self.energy[self.alive] -= self.energy_model.idle_cost(duty_cycle[self.alive])
        if control_packets > 0:
            live_idx = np.flatnonzero(self.alive)
            if len(live_idx):
                per_node = self.energy_model.control_cost(control_packets) / len(live_idx)
                self.energy[live_idx] -= per_node
        self.energy = np.maximum(0, self.energy)
        self.alive &= self.energy > self.energy_model.dead_threshold_j
        self.refresh_links()
        info = {
            "generated_packets": generated,
            "delivered_packets": int(delivered),
            "pdr": float(delivered / generated) if generated else 0.0,
            "alive_nodes": int(self.alive.sum()),
            "residual_energy_mean": float(self.energy[self.alive].mean()) if self.alive.any() else 0.0,
            "residual_energy_std": float(self.energy[self.alive].std()) if self.alive.any() else 0.0,
            "control_packets": int(control_packets),
            "retransmissions": int(retransmissions),
        }
        return self.state(), info

    def _follow_path(self, src: int, next_hop: np.ndarray):
        if not self.alive[src]:
            return None
        path = [src]
        seen = {src}
        cur = src
        for _ in range(self.max_hops):
            if self.sink_quality()[cur] >= 0.35:
                path.append(-1)
                return path
            nxt = int(next_hop[cur])
            if nxt < 0 or nxt >= self.n or not self.alive[nxt] or self.adjacency[cur, nxt] <= 0 or nxt in seen:
                return None
            path.append(nxt)
            seen.add(nxt)
            cur = nxt
        return None
