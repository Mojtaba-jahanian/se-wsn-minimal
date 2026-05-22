import numpy as np
import pandas as pd

def lifetime_metrics(round_logs: pd.DataFrame, n_nodes: int):
    df = round_logs.sort_values('round')
    alive = df['alive_nodes'].to_numpy()
    rounds = df['round'].to_numpy()
    def first_round(cond, default):
        idx = np.flatnonzero(cond)
        return int(rounds[idx[0]]) if len(idx) else int(default)
    fnd = first_round(alive < n_nodes, rounds[-1])
    hnd = first_round(alive <= n_nodes/2, rounds[-1])
    lnd = first_round(alive <= 1, rounds[-1])
    return fnd, hnd, lnd

def summarize_logs(logs: pd.DataFrame, n_nodes: int):
    rows = []
    for (seed, method), grp in logs.groupby(['seed', 'method']):
        fnd, hnd, lnd = lifetime_metrics(grp, n_nodes)
        generated = grp['generated_packets'].sum()
        delivered = grp['delivered_packets'].sum()
        pdr = delivered / generated if generated else 0.0
        rows.append({
            'seed': seed,
            'method': method,
            'FND': fnd,
            'HND': hnd,
            'LND': lnd,
            'PDR': pdr,
            'energy_dispersion_final': grp['residual_energy_std'].iloc[-1],
            'control_packets_mean': grp['control_packets'].mean(),
            'selected_next_hop_risk_mean': grp['selected_next_hop_risk'].mean(),
            'latency_sec_mean': grp['latency_sec'].mean(),
            'throughput_mean': grp['delivered_packets'].mean(),
        })
    return pd.DataFrame(rows)
