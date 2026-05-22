import pandas as pd

def mean_ci(df: pd.DataFrame, group_col='method'):
    rows = []
    metrics = ['FND','HND','LND','PDR','energy_dispersion_final','control_packets_mean','selected_next_hop_risk_mean','latency_sec_mean','throughput_mean']
    for method, grp in df.groupby(group_col):
        row = {group_col: method}
        n = len(grp)
        for m in metrics:
            if m in grp:
                mean = grp[m].mean()
                std = grp[m].std(ddof=1) if n > 1 else 0.0
                ci95 = 1.96 * std / (n ** 0.5) if n > 1 else 0.0
                row[m + '_mean'] = mean
                row[m + '_ci95'] = ci95
        rows.append(row)
    return pd.DataFrame(rows)
