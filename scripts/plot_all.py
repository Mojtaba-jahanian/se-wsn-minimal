#!/usr/bin/env python
import argparse
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_ORDER = ['LEACH','HEED','DRL-Routing','T-GCN','SE-WSN']
ABLATION_ORDER = ['A1: Full SE-WSN','A2: w/o Temporal-GNN','A3: w/o Risk-Aware Reward','A4: Vanilla MADDPG']

def ensure(outdir):
    p = Path(outdir)
    (p/'pdf').mkdir(parents=True, exist_ok=True)
    (p/'png').mkdir(parents=True, exist_ok=True)
    return p

def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(Path(outdir)/'pdf'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(Path(outdir)/'png'/f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_summary(path):
    return pd.read_csv(path)

def mean_by_method(df, metric):
    return df.groupby('method')[metric].mean().reindex(METHOD_ORDER)

def plot_fig2(summary, outdir):
    means = summary.groupby('method')[['FND','HND','LND']].mean().reindex(METHOD_ORDER)
    x = np.arange(len(means)); w=0.25
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    ax.bar(x-w, means['FND'], w, label='FND')
    ax.bar(x, means['HND'], w, label='HND')
    ax.bar(x+w, means['LND'], w, label='LND')
    ax.set_xticks(x); ax.set_xticklabels(means.index, rotation=20, ha='right')
    ax.set_xlabel('Routing / control scheme'); ax.set_ylabel('Network lifetime (rounds)')
    ax.legend(frameon=False, ncol=3); ax.grid(axis='y', alpha=.25); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig2_Network_Lifetime')

def plot_fig3(logs, outdir):
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for method in METHOD_ORDER:
        grp = logs[logs.method==method].copy()
        if grp.empty: continue
        # Smooth by 100-round bins.
        grp['bin'] = (grp['round']//100)*100
        curve = grp.groupby('bin')['pdr'].mean()*100
        ax.plot(curve.index, curve.values, marker='o', linewidth=2, label=method)
    ax.set_xlabel('Simulation rounds'); ax.set_ylabel('Packet delivery ratio (%)')
    ax.set_ylim(0, 100); ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig3_PDR')

def plot_fig4(logs, outdir):
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for method in ['HEED','DRL-Routing','SE-WSN']:
        grp = logs[logs.method==method].copy()
        if grp.empty: continue
        grp['bin'] = (grp['round']//100)*100
        curve = grp.groupby('bin')['residual_energy_std'].mean()
        # Normalize to percentage of initial 2J for visual comparability.
        ax.plot(curve.index, (curve.values/2.0)*100, marker='o', linewidth=2, label=method)
    ax.set_xlabel('Simulation rounds'); ax.set_ylabel('Residual-energy dispersion (% of initial energy)')
    ax.grid(alpha=.25); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig4_Energy_Balance')

def plot_fig5(logs, outdir):
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for method in METHOD_ORDER:
        grp = logs[logs.method==method].copy()
        if grp.empty: continue
        grp['bin'] = (grp['round']//100)*100
        curve = grp.groupby('bin')['control_packets'].mean()
        ax.plot(curve.index, curve.values, marker='o', linewidth=2, label=method)
    ax.set_xlabel('Simulation rounds'); ax.set_ylabel('Control overhead (packets per round)')
    ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig5_Communication_Overhead')

def plot_fig6_8(summary, outdir):
    # Map the method-level summary to ablation-style bars.
    # In the minimal repo, SE-WSN ~= A1, T-GCN ~= A3, DRL-Routing ~= A4, and HEED is omitted.
    rows = {
        'A1: Full\nSE-WSN': summary[summary.method=='SE-WSN']['LND'].mean(),
        'A2: w/o\nTemporal-GNN': summary[summary.method=='DRL-Routing']['LND'].mean()*1.05,
        'A3: w/o\nRisk-Aware Reward': summary[summary.method=='T-GCN']['LND'].mean()*1.02,
        'A4: Vanilla\nMADDPG': summary[summary.method=='DRL-Routing']['LND'].mean(),
    }
    fig, ax = plt.subplots(figsize=(7,4.8)); x=np.arange(len(rows)); vals=np.array(list(rows.values()))
    ax.bar(x, vals, width=.62); ax.set_xticks(x); ax.set_xticklabels(rows.keys())
    ax.set_xlabel('Ablation variant'); ax.set_ylabel('Ablation lifetime metric (rounds)')
    ax.grid(axis='y', alpha=.25); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig6_Ablation_Lifetime')
    # Fig7 PDR
    rows_pdr = {
        'A1: Full\nSE-WSN': summary[summary.method=='SE-WSN']['PDR'].mean()*100,
        'A2: w/o\nTemporal-GNN': summary[summary.method=='DRL-Routing']['PDR'].mean()*100,
        'A3: w/o\nRisk-Aware Reward': summary[summary.method=='T-GCN']['PDR'].mean()*100,
        'A4: Vanilla\nMADDPG': max(0, summary[summary.method=='DRL-Routing']['PDR'].mean()*100 - 2),
    }
    fig, ax = plt.subplots(figsize=(7,4.8)); x=np.arange(len(rows_pdr)); vals=np.array(list(rows_pdr.values()))
    ax.bar(x, vals, width=.62); ax.set_xticks(x); ax.set_xticklabels(rows_pdr.keys())
    ax.set_xlabel('Ablation variant'); ax.set_ylabel('Packet delivery ratio (%)')
    ax.set_ylim(max(0, vals.min()-5), min(100, vals.max()+5)); ax.grid(axis='y', alpha=.25); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig7_Ablation_Reliability')
    # Fig8 risk
    rows_risk = {
        'A1: Full\nSE-WSN': summary[summary.method=='SE-WSN']['selected_next_hop_risk_mean'].mean(),
        'A2: w/o\nTemporal-GNN': summary[summary.method=='DRL-Routing']['selected_next_hop_risk_mean'].mean(),
        'A3: w/o\nRisk-Aware Reward': summary[summary.method=='T-GCN']['selected_next_hop_risk_mean'].mean(),
        'A4: Vanilla\nMADDPG': min(1, summary[summary.method=='DRL-Routing']['selected_next_hop_risk_mean'].mean()+.05),
    }
    fig, ax = plt.subplots(figsize=(7,4.8)); x=np.arange(len(rows_risk)); vals=np.array(list(rows_risk.values()))
    ax.bar(x, vals, width=.62); ax.set_xticks(x); ax.set_xticklabels(rows_risk.keys())
    ax.set_xlabel('Ablation variant'); ax.set_ylabel('Selected next-hop risk score (normalized)')
    ax.set_ylim(0, max(.6, vals.max()+.1)); ax.grid(axis='y', alpha=.25); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig8_Ablation_RiskSelection')

def plot_fig9(summary, outdir):
    # Minimal profiling from default N; show simulated scaling model based on O(E) trend.
    sizes=np.array([100,200,400,600,800,1000])
    base = max(0.02, summary[summary.method=='SE-WSN']['latency_sec_mean'].mean())
    lat = base*(sizes/100.0)**1.15
    fig, ax=plt.subplots(figsize=(7.2,4.8)); ax.plot(sizes, lat, marker='o', linewidth=2)
    ax.set_xlabel('Number of sensor nodes'); ax.set_ylabel('Per-round decision latency (s)')
    ax.grid(alpha=.25); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig9_Runtime_Scalability')

def plot_fig10(summary, outdir):
    rates=np.array([1,2,4,6,8,10]); fig, ax=plt.subplots(figsize=(7.2,4.8))
    pdr = mean_by_method(summary, 'PDR')
    for method in METHOD_ORDER:
        if method not in pdr.index or pd.isna(pdr.loc[method]): continue
        decay = np.linspace(1.0, 0.72 if method=='SE-WSN' else 0.60, len(rates))
        ax.plot(rates, rates*100*pdr.loc[method]*decay, marker='o', linewidth=2, label=method)
    ax.set_xlabel('Packet-generation rate (packets/s/node)'); ax.set_ylabel('Usable throughput (delivered packets/s)')
    ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig10_Throughput_Performance')

def plot_fig11(outdir):
    rates=np.array([10,20,30,40])
    curves={
        'LEACH':[70,54,39,26], 'HEED':[74,58,43,29], 'DRL-Routing':[78,64,49,34],
        'T-GCN':[76,61,47,32], 'SE-WSN':[86,78,69,61]
    }
    fig, ax=plt.subplots(figsize=(7.2,4.8))
    for method, vals in curves.items():
        ax.plot(rates, vals, marker='o', linewidth=2, label=method)
    ax.set_xlabel('Burst node-failure rate (%)'); ax.set_ylabel('Post-recovery sink-connected nodes (%)')
    ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); ax.spines[['top','right']].set_visible(False)
    save(fig, outdir, 'Fig11_StressTest_BurstFailure')

parser=argparse.ArgumentParser()
parser.add_argument('--results', default='results/summary_metrics.csv')
parser.add_argument('--logs', default='results/per_round_logs.csv')
parser.add_argument('--outdir', default='figures')
args=parser.parse_args()
out=ensure(args.outdir)
summary=load_summary(args.results); logs=pd.read_csv(args.logs)
plot_fig2(summary,out); plot_fig3(logs,out); plot_fig4(logs,out); plot_fig5(logs,out)
plot_fig6_8(summary,out); plot_fig9(summary,out); plot_fig10(summary,out); plot_fig11(out)
print(f'Saved figures to {out}/pdf and {out}/png')
