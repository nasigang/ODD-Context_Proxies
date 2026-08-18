#!/usr/bin/env python3
"""
Construct Validity Analysis
==============================
Compares diagnostic vectors against external KPIs to establish
convergent, discriminant, and criterion-related validity.

Score vectors compared:
  D_s^risk  — peak of R_t curve
  D_s^tail  — mean S_t (surprise) across event frames
  B_s       — max_event_duration (response vector)
  External KPIs — max_jerk, DRAC, RSS margin, etc.

Analysis:
  - Spearman / Kendall rank correlations
  - Scenario-block bootstrap 95% CI (1000 resamples)
  - ODD-stratified analysis
  - Fixed FPR event discrimination
  - Coverage report

Usage:
    python phase2_womd/construct_validity.py --max-scenarios 20
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
FPR_LEVELS = [0.05, 0.10]


# ---------------------------------------------------------------------------
# Score vector extraction
# ---------------------------------------------------------------------------

def extract_score_vectors(
    df_diag: pd.DataFrame,
    df_curves: pd.DataFrame,
) -> pd.DataFrame:
    """Extract D_s^risk, D_s^tail, B_s per scenario."""
    result = df_diag[["scenario_id"]].copy()

    # D_s^risk = peak
    result["Ds_risk"] = df_diag["peak"].values

    # B_s = max_event_duration (response vector)
    result["Bs_response"] = df_diag["max_event_duration"].values

    # D_s^tail = mean S_t across event frames (where S_t is finite)
    if df_curves is not None and not df_curves.empty and "St" in df_curves.columns:
        tail_means = []
        for sid in result["scenario_id"]:
            sc_curves = df_curves[df_curves["scenario_id"] == sid]
            st_vals = sc_curves["St"].dropna()
            tail_means.append(float(st_vals.mean()) if len(st_vals) > 0 else np.nan)
        result["Ds_tail"] = tail_means
    else:
        result["Ds_tail"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def pairwise_correlations(
    scores: pd.DataFrame,
    score_cols: List[str],
    kpi_cols: List[str],
) -> pd.DataFrame:
    """Compute Spearman + Kendall between all score×KPI pairs."""
    rows = []
    for scol in score_cols:
        for kcol in kpi_cols:
            x = scores[scol].values if scol in scores.columns else np.full(len(scores), np.nan)
            y = scores[kcol].values if kcol in scores.columns else np.full(len(scores), np.nan)
            valid = np.isfinite(x) & np.isfinite(y)

            if valid.sum() < 3:
                rows.append({
                    "score": scol, "kpi": kcol,
                    "spearman": np.nan, "kendall": np.nan,
                    "n_valid": int(valid.sum()),
                })
                continue

            sp, sp_p = spearmanr(x[valid], y[valid])
            kt, kt_p = kendalltau(x[valid], y[valid])

            rows.append({
                "score": scol, "kpi": kcol,
                "spearman": round(float(sp), 4),
                "spearman_p": round(float(sp_p), 6),
                "kendall": round(float(kt), 4),
                "kendall_p": round(float(kt_p), 6),
                "n_valid": int(valid.sum()),
            })

    return pd.DataFrame(rows)


def bootstrap_correlations(
    scores: pd.DataFrame,
    score_cols: List[str],
    kpi_cols: List[str],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Scenario-block bootstrap 95% CI for Spearman correlations."""
    rng = np.random.RandomState(seed)
    rows = []

    for scol in score_cols:
        for kcol in kpi_cols:
            x = scores[scol].values if scol in scores.columns else np.full(len(scores), np.nan)
            y = scores[kcol].values if kcol in scores.columns else np.full(len(scores), np.nan)
            valid = np.isfinite(x) & np.isfinite(y)
            n = int(valid.sum())

            if n < 3:
                rows.append({
                    "score": scol, "kpi": kcol,
                    "ci_lower": np.nan, "ci_upper": np.nan, "n": n,
                })
                continue

            xv, yv = x[valid], y[valid]
            boot_corrs = []
            for _ in range(n_boot):
                idx = rng.choice(n, size=n, replace=True)
                sp, _ = spearmanr(xv[idx], yv[idx])
                if np.isfinite(sp):
                    boot_corrs.append(sp)

            if boot_corrs:
                rows.append({
                    "score": scol, "kpi": kcol,
                    "ci_lower": round(float(np.percentile(boot_corrs, 2.5)), 4),
                    "ci_upper": round(float(np.percentile(boot_corrs, 97.5)), 4),
                    "boot_mean": round(float(np.mean(boot_corrs)), 4),
                    "n": n,
                })
            else:
                rows.append({
                    "score": scol, "kpi": kcol,
                    "ci_lower": np.nan, "ci_upper": np.nan, "n": n,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ODD-stratified analysis
# ---------------------------------------------------------------------------

def odd_stratified_analysis(
    scores: pd.DataFrame,
    df_model: Optional[pd.DataFrame],
    score_cols: List[str],
    kpi_cols: List[str],
) -> Dict:
    """Spearman correlations stratified by ODD features."""
    result = {}
    if df_model is None or df_model.empty:
        return result

    # Join ODD features
    merged = scores.merge(
        df_model.groupby("scenario_id").first().reset_index()
        [["scenario_id", "odd_n_valid_agents", "odd_has_signal"]],
        on="scenario_id", how="left",
    )

    # Stratify by traffic
    if "odd_n_valid_agents" in merged.columns:
        med = merged["odd_n_valid_agents"].median()
        for label, mask in [("low_traffic", merged["odd_n_valid_agents"] <= med),
                            ("high_traffic", merged["odd_n_valid_agents"] > med)]:
            sub = merged[mask]
            if len(sub) < 5:
                continue
            corrs = pairwise_correlations(sub, score_cols, kpi_cols)
            result[label] = corrs.to_dict(orient="records")

    # Stratify by signal
    if "odd_has_signal" in merged.columns:
        for label, val in [("no_signal", 0), ("has_signal", 1)]:
            sub = merged[merged["odd_has_signal"] == val]
            if len(sub) < 5:
                continue
            corrs = pairwise_correlations(sub, score_cols, kpi_cols)
            result[label] = corrs.to_dict(orient="records")

    return result


# ---------------------------------------------------------------------------
# Event discrimination
# ---------------------------------------------------------------------------

def event_discrimination(
    scores: pd.DataFrame,
    score_col: str,
    kpi_col: str,
    fpr_levels: List[float] = FPR_LEVELS,
) -> Dict:
    """Fixed FPR event discrimination: sensitivity for top-quintile KPI."""
    x = scores[score_col].values if score_col in scores.columns else np.full(len(scores), np.nan)
    y = scores[kpi_col].values if kpi_col in scores.columns else np.full(len(scores), np.nan)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 10:
        return {}

    xv, yv = x[valid], y[valid]

    # Define "events" as top quintile of KPI
    threshold = np.percentile(yv, 80)
    is_event = yv >= threshold

    result = {}
    for fpr in fpr_levels:
        # Score threshold at given FPR
        n_neg = (~is_event).sum()
        n_fp = int(fpr * n_neg)
        neg_scores = np.sort(xv[~is_event])[::-1]
        if n_fp < len(neg_scores) and n_fp > 0:
            score_thresh = neg_scores[n_fp - 1]
        else:
            score_thresh = np.percentile(xv, 100 * (1 - fpr))

        # Sensitivity (TPR)
        tp = (xv[is_event] >= score_thresh).sum()
        tpr = float(tp / is_event.sum()) if is_event.sum() > 0 else np.nan

        result[f"FPR_{fpr:.2f}"] = {
            "sensitivity": round(tpr, 4),
            "score_threshold": round(float(score_thresh), 4),
            "n_events": int(is_event.sum()),
            "n_non_events": int((~is_event).sum()),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_construct_validity(
    output_root: str,
    max_scenarios: Optional[int] = None,
) -> Dict:
    """Run full construct validity analysis."""
    os.makedirs(output_root, exist_ok=True)

    # Load data
    diag_path = os.path.join(output_root, "scenario_diagnostic_vector.parquet")
    curve_path = os.path.join(output_root, "diagnostic_figure_data.parquet")
    kpi_path = os.path.join(output_root, "external_kpi.parquet")
    model_path = os.path.join(output_root, "model_table.parquet")

    # Compute external KPIs if not present
    if not os.path.isfile(kpi_path):
        print("[KPI] Computing external KPIs...")
        import pyarrow.parquet as pq
        parquet_root = os.path.join(output_root, "parquet")
        df_agent = pq.read_table(os.path.join(parquet_root, "agent_state")).to_pandas()
        df_map = pq.read_table(os.path.join(parquet_root, "map_feature")).to_pandas()

        pair_path = os.path.join(output_root, "pair_metrics.parquet")
        df_pairs = pd.read_parquet(pair_path) if os.path.isfile(pair_path) else pd.DataFrame()

        sids = df_agent["scenario_id"].unique()
        if max_scenarios:
            sids = sids[:max_scenarios]

        from phase2_womd.external_kpi import compute_all_kpis
        df_kpi, coverage = compute_all_kpis(df_agent, df_pairs, df_map, list(sids))
        df_kpi.to_parquet(kpi_path, index=False)

        cov_path = os.path.join(output_root, "external_kpi_coverage.json")
        with open(cov_path, "w") as f:
            json.dump(coverage, f, indent=2, default=str)
    else:
        df_kpi = pd.read_parquet(kpi_path)

    # Load diagnostics
    if not os.path.isfile(diag_path):
        print("[DIAG] Computing diagnostics...")
        from phase2_womd.scenario_diagnostics import compute_all_diagnostics
        pred_path = os.path.join(output_root, "frame_predictions.parquet")
        df_pred = pd.read_parquet(pred_path)
        if max_scenarios:
            keep = df_pred["scenario_id"].unique()[:max_scenarios]
            df_pred = df_pred[df_pred["scenario_id"].isin(keep)]
        df_diag, df_curves = compute_all_diagnostics(df_pred)
        df_diag.to_parquet(diag_path, index=False)
        df_curves.to_parquet(curve_path, index=False)
    else:
        df_diag = pd.read_parquet(diag_path)
        df_curves = pd.read_parquet(curve_path) if os.path.isfile(curve_path) else pd.DataFrame()

    df_model = pd.read_parquet(model_path) if os.path.isfile(model_path) else pd.DataFrame()

    print(f"[Load] {len(df_diag)} diagnostic vectors, {len(df_kpi)} KPI vectors")

    # Extract score vectors
    scores = extract_score_vectors(df_diag, df_curves)

    # Merge KPIs
    kpi_cols = ["max_jerk", "drac_max", "rss_margin_min",
                "cross_track_error_max", "ttlc_min", "offroad_ratio"]
    scores = scores.merge(df_kpi[["scenario_id"] + kpi_cols], on="scenario_id", how="left")

    score_cols = ["Ds_risk", "Ds_tail", "Bs_response"]

    # --- Correlations ---
    print("\n[Correlations]")
    corr_df = pairwise_correlations(scores, score_cols, kpi_cols)
    print(corr_df.to_string(index=False))

    # --- Bootstrap CI ---
    print("\n[Bootstrap CI]")
    boot_df = bootstrap_correlations(scores, score_cols, kpi_cols)

    # --- ODD stratified ---
    print("\n[ODD Stratified]")
    odd_results = odd_stratified_analysis(scores, df_model, score_cols, kpi_cols)

    # --- Event discrimination ---
    print("\n[Event Discrimination]")
    disc_results = {}
    for kcol in kpi_cols:
        disc = event_discrimination(scores, "Ds_risk", kcol, FPR_LEVELS)
        if disc:
            disc_results[kcol] = disc

    # Save outputs
    corr_path = os.path.join(output_root, "construct_validity_metrics.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\n[OK] {corr_path}")

    boot_path = os.path.join(output_root, "bootstrap_results.csv")
    boot_df.to_csv(boot_path, index=False)
    print(f"[OK] {boot_path}")

    report = {
        "correlations": corr_df.to_dict(orient="records"),
        "bootstrap_ci": boot_df.to_dict(orient="records"),
        "odd_stratified": odd_results,
        "event_discrimination": disc_results,
        "n_scenarios": int(len(scores)),
    }
    report_path = os.path.join(output_root, "construct_validity_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[OK] {report_path}")

    return report


def main():
    ap = argparse.ArgumentParser(description="Construct Validity Analysis")
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    run_construct_validity(args.output_root, args.max_scenarios)


if __name__ == "__main__":
    main()
