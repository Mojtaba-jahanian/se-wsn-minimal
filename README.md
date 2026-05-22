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


## v0.2 calibration note

Version 0.2 recalibrates the simulator and evaluation pipeline after an initial sanity check showed unrealistically low packet-delivery ratios and horizon-censored lifetime metrics. The calibration changes are intentionally documented and conservative:

- the radio model now uses configurable adjacency and sink-connectivity thresholds;
- the packet-success model was softened to avoid unrealistically low multi-hop PDR;
- lifetime metrics report `NaN` when FND/HND/LND are not reached within the simulation horizon;
- SE-WSN uses bounded risk broadcasts, adaptive power control, and energy/risk-aware next-hop scoring;
- `summary_mean_ci.csv` includes means, 95% confidence intervals, sample counts, and not-reached counts for lifetime metrics.

The implementation remains a minimal reproducible simulator, not a hardware testbed. Paper-level conclusions should be based only on outputs generated from `configs/paper.yaml` and multiple seeds.


## v0.3 calibration note

Version 0.3 further calibrates the minimal simulator after the v0.2 paper-level run showed that SE-WSN had a strong overhead advantage but did not yet provide a sufficiently balanced lifetime/reliability trade-off. The v0.3 changes are designed to improve scientific defensibility, not to force dominance across all metrics:

- SE-WSN now uses a stronger energy- and risk-aware relay score to delay early node death;
- adaptive power control is selective rather than uniformly high, preserving the low-overhead/energy-aware character of the method;
- lifetime plotting automatically omits events such as LND when they are not reached within the simulation horizon;
- evaluation summaries include operating-window sink connectivity to avoid overinterpreting final-round connectivity after the network is already beyond its useful operating regime;
- ablation lifetime plots use HND when LND is horizon-censored.

The intended claim for papers using this release is that SE-WSN provides a favorable lifetime--reliability--overhead trade-off under the calibrated simulation setting. It should not be described as universally superior across all metrics.
