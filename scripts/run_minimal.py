#!/usr/bin/env python
import argparse
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import yaml
from sewsn.evaluation.run_experiment import run_experiment
from sewsn.evaluation.aggregate_results import mean_ci

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
parser.add_argument('--outdir', default='results')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
logs, summary = run_experiment(cfg, args.seeds)
logs.to_csv(out / 'per_round_logs.csv', index=False)
summary.to_csv(out / 'summary_metrics.csv', index=False)
mean_ci(summary).to_csv(out / 'summary_mean_ci.csv', index=False)
print('\nSaved:')
print(out / 'per_round_logs.csv')
print(out / 'summary_metrics.csv')
print(out / 'summary_mean_ci.csv')
