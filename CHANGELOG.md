# Changelog

## v0.2.0 - Calibrated minimal reproducible release

- Added calibrated radio thresholds for adjacency and sink connectivity.
- Adjusted packet-success model to avoid unrealistically low multi-hop PDR.
- Added `sink_connected_ratio` to per-round logs.
- Updated lifetime metrics to report `NaN` when FND/HND/LND are not reached within the horizon.
- Added not-reached counts to `summary_mean_ci.csv`.
- Recalibrated SE-WSN policy with risk-aware next-hop selection, adaptive power, bounded risk broadcasts, and duty-cycle regulation.
- Updated default, quick, and paper configurations for reproducible calibrated runs.

## v0.3.0 - Balanced calibrated release

- Rebalanced the SE-WSN controller toward energy-aware, risk-aware relay selection.
- Added stronger documentation that the expected contribution is a trade-off, not universal metric dominance.
- Added operating-window sink-connectivity summaries.
- Updated plotting to omit horizon-censored lifetime events and to use HND for ablation lifetime when LND is unavailable.
- Slightly recalibrated radio and energy settings for paper-level runs.
