# SE-WSN Minimal Reproducible Implementation

This repository contains a compact, reproducible implementation of a **Self-Evolving Wireless Sensor Network (SE-WSN)** experiment pipeline. It is designed to support a journal submission by producing real experiment logs, summary tables, and publication-ready figures from code rather than reconstructed plots.

The implementation is intentionally minimal: it provides a controlled WSN simulator, temporal graph construction, risk labeling, a lightweight Temporal-GNN module, a risk-aware SE-WSN controller, and four baselines: LEACH, HEED, DRL-Routing, and T-GCN-Routing.

## What this repository is for

- Generate reproducible per-seed logs for lifetime, PDR, energy balance, overhead, scalability, throughput, and stress tests.
- Export `results/summary_metrics.csv` and figure PDFs/PNGs for the paper.
- Provide transparent code for reviewers and editors.

## Important note

This is a **minimal reproducible simulator**. It is not a hardware testbed and should not be described as field deployment. For a high-impact journal submission, report results as simulation-based and discuss the limitations of the simplified channel, energy, and control models.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_minimal.py --config configs/default.yaml --seeds 0 1 2
python scripts/plot_all.py --results results/summary_metrics.csv --outdir figures
```

For the paper version, use ten seeds:

```bash
python scripts/run_minimal.py --config configs/default.yaml --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/plot_all.py --results results/summary_metrics.csv --outdir figures
```

## Outputs

- `results/per_round_logs.csv`: round-level logs.
- `results/summary_metrics.csv`: method-level metrics per seed.
- `figures/pdf/Fig2_Network_Lifetime.pdf` through `Fig11_StressTest.pdf`.
- `figures/png/*.png`: preview versions.

## Repository structure

```text
sewsn/
  env/          WSN simulator, energy/link/failure models
  graph/        temporal graph builder and risk labeling
  models/       Temporal-GNN and MADDPG-lite modules
  baselines/    LEACH, HEED, DRL-Routing, T-GCN-Routing, SE-WSN controllers
  evaluation/   metrics, experiment runner, aggregation
scripts/        command-line experiment and plotting scripts
configs/        YAML configuration files
```

## Recommended paper wording

> The code, configuration files, and figure-generation scripts are released for reproducibility. All figures are generated from per-seed simulator logs using the public scripts. The evaluation should be interpreted as real-data-calibrated or synthetic simulation evidence rather than hardware deployment.

## Citation / license

Add your paper citation here after acceptance. The code can be released under MIT license or another license required by your institution.
