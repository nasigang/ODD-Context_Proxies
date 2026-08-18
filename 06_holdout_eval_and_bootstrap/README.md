# 06. Sealed Holdout Evaluation and Statistical Inference

Contains the evaluation engine, scenario-block bootstrap framework, and precision-recall metric calculators.

## Inference Protocol
- **Evaluation Set**: Sealed holdout cohort ($N=255,164$ frames, $2,804$ scenarios).
- **Bootstrap**: Paired scenario-block percentile bootstrap ($B=1,000$ iterations) preserving intra-scenario temporal autocorrelation.
- **Key Metrics**:
  - Average Precision (AP / PR-AUC)
  - AUROC
  - Brier Calibration Score
  - Incremental Delta: $\Delta\text{AP} = \text{AP}(M_{P+E_{\text{all}}}) - \text{AP}(M_P) = +0.0147$ ($95\%$ CI $[+0.0005, +0.0285]$).
