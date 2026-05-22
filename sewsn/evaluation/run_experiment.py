import time
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd
from sewsn.env.wsn_env import WSNEnv
from sewsn.graph.temporal_graph_builder import TemporalGraphBuilder
from sewsn.models.temporal_gnn import RiskEstimator
from sewsn.baselines import LEACHPolicy, HEEDPolicy, DRLRoutingPolicy, TGCNRoutingPolicy, SEWSNPolicy
from sewsn.evaluation.metrics import summarize_logs

POLICY_MAP = {
    'LEACH': LEACHPolicy,
    'HEED': HEEDPolicy,
    'DRL-Routing': DRLRoutingPolicy,
    'T-GCN': TGCNRoutingPolicy,
    'SE-WSN': SEWSNPolicy,
}

def make_policy(method: str, cfg: Dict[str, Any]):
    if method == 'LEACH':
        return LEACHPolicy(ch_prob=cfg.get('control', {}).get('leach_ch_prob', 0.05), recluster_interval=cfg.get('control', {}).get('recluster_interval', 200))
    if method == 'SE-WSN':
        r = cfg.get('risk', {})
        c = cfg.get('control', {})
        return SEWSNPolicy(r.get('node_fail_threshold', 0.70), r.get('link_degrade_threshold', 0.65), c.get('risk_broadcast_interval', 10))
    return POLICY_MAP[method]()

def compute_risk(env: WSNEnv, estimator: RiskEstimator):
    degree = env.adjacency.sum(axis=1)
    mean_lq = env.link_quality.sum(axis=1) / np.maximum(degree, 1.0)
    energy_norm = env.energy / env.energy_model.initial_j
    node_risk = estimator.node_failure_risk(energy_norm, mean_lq, env.traffic, degree)
    link_risk = estimator.link_degradation_risk(env.link_quality, energy_norm)
    cluster_risk = estimator.cluster_instability(energy_norm, env.alive)
    return {'node_risk': node_risk, 'link_risk': link_risk, 'cluster_risk': cluster_risk}

def run_single(cfg: Dict[str, Any], seed: int, method: str):
    env = WSNEnv(cfg, seed=seed)
    policy = make_policy(method, cfg)
    policy.reset()
    tg = TemporalGraphBuilder(cfg.get('risk', {}).get('temporal_window', 4))
    estimator = RiskEstimator(
        energy_dead_threshold=cfg.get('energy', {}).get('dead_threshold_j', 0.05),
        link_quality_min=cfg.get('risk', {}).get('link_quality_min', 0.45),
    )
    rows = []
    rounds = int(cfg['experiment'].get('rounds', 2200))
    for _ in range(rounds):
        features = env.features()
        tg.add(env.round_id, features, env.adjacency, env.link_quality)
        t0 = time.perf_counter()
        risk = compute_risk(env, estimator)
        action = policy.act(env, risk=risk)
        latency = time.perf_counter() - t0
        selected_risk = []
        for i in np.flatnonzero(env.alive):
            j = int(action.next_hop[i])
            if 0 <= j < env.n:
                selected_risk.append(float(risk['node_risk'][j]))
        _, info = env.step(action)
        info.update({
            'seed': seed,
            'method': method,
            'round': env.round_id,
            'selected_next_hop_risk': float(np.mean(selected_risk)) if selected_risk else 1.0,
            'latency_sec': float(latency),
            'cluster_risk': float(risk['cluster_risk']),
        })
        rows.append(info)
        if env.alive.sum() <= 1:
            break
    return pd.DataFrame(rows)

def run_experiment(cfg: Dict[str, Any], seeds: Iterable[int]):
    methods = cfg['experiment'].get('methods', list(POLICY_MAP))
    all_logs = []
    for seed in seeds:
        for method in methods:
            print(f'Running seed={seed} method={method}')
            all_logs.append(run_single(cfg, int(seed), method))
    logs = pd.concat(all_logs, ignore_index=True)
    summary = summarize_logs(logs, int(cfg['experiment'].get('n_nodes', 100)))
    return logs, summary
