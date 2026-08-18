#!/usr/bin/env python3
"""
Temporal Robustness Analysis
==============================
Evaluates stability of scenario diagnostic rankings under perturbations:
  - Temporal resampling (5 Hz, 2 Hz)
  - Random frame dropout (5%, 10%)
  - Threshold sensitivity (θ_on = q90, q95, q99 of R_t)

Metrics:
  - Spearman & Kendall rank correlations
  - Top-k overlap (k=10, 20, 50)
  - ICC (intraclass correlation coefficient)
  - Component-wise relative error
  - Scenario-block bootstrap CI (1000 resamples)

Usage:
    python phase2_womd/temporal_robustness.py --max-scenarios 20
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

from phase2_womd.diag_config import DiagConfig
from phase2_womd.scenario_diagnostics import compute_all_diagnostics

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
TOP_K_VALUES = [10, 20, 50]

# Primary D_s components for ranking
RANK_COMPONENTS = [
    "peak", "exceedance_duration", "event_count",
    "max_event_duration", "recovery_time", "predictive_uncertainty",
]


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def subsample_hz(df: pd.DataFrame, target_hz: float,
                 original_hz: float = 10.0) -> pd.DataFrame:
    """Subsample frames to target frequency."""
    step = max(1, int(round(original_hz / target_hz)))
    groups = []
    for sid, grp in df.groupby("scenario_id"):
        grp_sorted = grp.sort_values("time_index")
        indices = list(range(0, len(grp_sorted), step))
        groups.append(grp_sorted.iloc[indices])
    return pd.concat(groups, ignore_index=True) if groups else df.iloc[:0]


def random_dropout(df: pd.DataFrame, dropout_rate: float,
                   seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Randomly drop frames at given rate, preserving scenario structure."""
    rng = np.random.RandomState(seed)
    keep_mask = rng.random(len(df)) >= dropout_rate
    # Always keep first and last frame of each scenario
    for sid, grp in df.groupby("scenario_id"):
        idx = grp.index
        keep_mask[idx[0]] = True
        keep_mask[idx[-1]] = True
    return df[keep_mask].reset_index(drop=True)


def threshold_from_quantile(
    df_pred: pd.DataFrame,
    quantile: float,
    cfg: DiagConfig,
) -> DiagConfig:
    """Create a new config with theta_on set from R_t quantile."""
    from phase2_womd.scenario_diagnostics import compute_Rt

    Rt_all = compute_Rt(
        df_pred["pred_p_exposure"].values,
        df_pred["pred_mu"].values,
        df_pred["pred_sigma"].values,
        tau=cfg.tau_primary,
    )
    theta_on = float(np.quantile(Rt_all[np.isfinite(Rt_all)], quantile))
    theta_off = theta_on * 0.5  # maintain ratio

    if theta_on <= theta_off:
        theta_off = theta_on * 0.4

    return DiagConfig(
        theta_on=max(theta_on, 0.01),
        theta_off=max(theta_off, 0.005),
        minimum_event_duration_s=cfg.minimum_event_duration_s,
        merge_gap_s=cfg.merge_gap_s,
        recovery_hold_time_s=cfg.recovery_hold_time_s,
        epsilon=cfg.epsilon,
        tau_thresholds=cfg.tau_thresholds,
        tau_primary=cfg.tau_primary,
    )


# ---------------------------------------------------------------------------
# Ranking stability metrics
# ---------------------------------------------------------------------------

def compute_rank_correlation(
    ref_scores: np.ndarray,
    pert_scores: np.ndarray,
) -> Dict:
    """Compute Spearman and Kendall rank correlations."""
    valid = np.isfinite(ref_scores) & np.isfinite(pert_scores)
    if valid.sum() < 3:
        return {"spearman": np.nan, "kendall": np.nan}

    sp_r, sp_p = spearmanr(ref_scores[valid], pert_scores[valid])
    kt_r, kt_p = kendalltau(ref_scores[valid], pert_scores[valid])

    return {
        "spearman": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 6),
        "kendall": round(float(kt_r), 4),
        "kendall_p": round(float(kt_p), 6),
    }


def compute_top_k_overlap(
    ref_ids: np.ndarray,
    ref_scores: np.ndarray,
    pert_ids: np.ndarray,
    pert_scores: np.ndarray,
    k_values: List[int],
) -> Dict:
    """Compute top-k overlap between reference and perturbed rankings."""
    result = {}
    for k in k_values:
        if len(ref_scores) < k or len(pert_scores) < k:
            result[f"top_{k}"] = np.nan
            continue

        ref_order = np.argsort(-ref_scores)[:k]
        pert_order = np.argsort(-pert_scores)[:k]

        ref_top = set(ref_ids[ref_order])
        pert_top = set(pert_ids[pert_order])

        overlap = len(ref_top & pert_top) / k
        result[f"top_{k}"] = round(float(overlap), 4)

    return result


def compute_icc(
    scores_matrix: np.ndarray,
) -> float:
    """Compute ICC(2,1) — two-way random, single measures.

    scores_matrix: (n_scenarios, n_conditions) array
    """
    n, k = scores_matrix.shape
    if n < 2 or k < 2:
        return np.nan

    # row means, col means, grand mean
    row_means = scores_matrix.mean(axis=1)
    col_means = scores_matrix.mean(axis=0)
    grand_mean = scores_matrix.mean()

    # Sum of squares
    ss_total = np.sum((scores_matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_cols = ss_cols / (k - 1)

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-12:
        return np.nan

    icc = (ms_rows - ms_error) / denom
    return round(float(icc), 4)


def compute_component_relative_error(
    ref_vecs: pd.DataFrame,
    pert_vecs: pd.DataFrame,
    components: List[str],
) -> Dict:
    """Compute per-component relative error between reference and perturbed."""
    result = {}
    for comp in components:
        if comp not in ref_vecs.columns or comp not in pert_vecs.columns:
            continue
        ref = ref_vecs[comp].values.astype(float)
        pert = pert_vecs[comp].values.astype(float)

        denom = np.maximum(np.abs(ref), 1e-12)
        rel_err = np.abs(ref - pert) / denom
        valid = np.isfinite(rel_err)

        if valid.sum() > 0:
            result[comp] = {
                "mean_rel_error": round(float(rel_err[valid].mean()), 4),
                "median_rel_error": round(float(np.median(rel_err[valid])), 4),
                "max_rel_error": round(float(rel_err[valid].max()), 4),
            }
    return result


def bootstrap_ci(
    ref_scores: np.ndarray,
    pert_scores: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Scenario-block bootstrap CI for Spearman correlation."""
    rng = np.random.RandomState(seed)
    n = len(ref_scores)
    if n < 3:
        return {"ci_lower": np.nan, "ci_upper": np.nan}

    boot_corrs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r_ref = ref_scores[idx]
        r_pert = pert_scores[idx]
        valid = np.isfinite(r_ref) & np.isfinite(r_pert)
        if valid.sum() >= 3:
            sp, _ = spearmanr(r_ref[valid], r_pert[valid])
            boot_corrs.append(sp)

    if not boot_corrs:
        return {"ci_lower": np.nan, "ci_upper": np.nan}

    return {
        "ci_lower": round(float(np.percentile(boot_corrs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_corrs, 97.5)), 4),
        "mean": round(float(np.mean(boot_corrs)), 4),
    }


# ---------------------------------------------------------------------------
# Main robustness pipeline
# ---------------------------------------------------------------------------

def run_robustness_analysis(
    df_pred: pd.DataFrame,
    output_root: str,
    cfg: Optional[DiagConfig] = None,
) -> Dict:
    """Run all robustness experiments."""
    if cfg is None:
        cfg = DiagConfig()

    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("TEMPORAL ROBUSTNESS ANALYSIS")
    print("=" * 60)

    # Reference (original)
    print("\n[REF] Computing reference diagnostics...")
    ref_vecs, ref_curves = compute_all_diagnostics(df_pred, cfg)
    ref_vecs = ref_vecs.sort_values("scenario_id").reset_index(drop=True)

    # Define perturbation experiments
    experiments = {
        "original": {"desc": "Full 10 Hz", "df": df_pred, "cfg": cfg},
        "5hz": {"desc": "Subsampled 5 Hz", "df": subsample_hz(df_pred, 5.0), "cfg": cfg},
        "2hz": {"desc": "Subsampled 2 Hz", "df": subsample_hz(df_pred, 2.0), "cfg": cfg},
        "dropout_5pct": {"desc": "5% random dropout", "df": random_dropout(df_pred, 0.05), "cfg": cfg},
        "dropout_10pct": {"desc": "10% random dropout", "df": random_dropout(df_pred, 0.10), "cfg": cfg},
    }

    # Threshold sensitivity
    for q_label, q_val in [("q90", 0.90), ("q95", 0.95), ("q99", 0.99)]:
        try:
            q_cfg = threshold_from_quantile(df_pred, q_val, cfg)
            experiments[f"theta_{q_label}"] = {
                "desc": f"θ_on from {q_label} of R_t",
                "df": df_pred,
                "cfg": q_cfg,
            }
        except Exception as e:
            print(f"  [WARN] Skipping {q_label}: {e}")

    # Run experiments
    all_results = {}
    score_matrix_list = []
    exp_names = []

    ref_ids = ref_vecs["scenario_id"].values
    ref_peak = ref_vecs["peak"].values

    for exp_name, exp in experiments.items():
        print(f"\n[{exp_name}] {exp['desc']}...")
        exp_vecs, _ = compute_all_diagnostics(exp["df"], exp["cfg"])

        # Align with reference scenarios
        exp_vecs = exp_vecs.sort_values("scenario_id").reset_index(drop=True)
        common = set(ref_ids) & set(exp_vecs["scenario_id"].values)

        if len(common) < 3:
            print(f"  [SKIP] Only {len(common)} common scenarios")
            continue

        ref_mask = ref_vecs["scenario_id"].isin(common)
        exp_mask = exp_vecs["scenario_id"].isin(common)
        ref_sub = ref_vecs[ref_mask].sort_values("scenario_id").reset_index(drop=True)
        exp_sub = exp_vecs[exp_mask].sort_values("scenario_id").reset_index(drop=True)

        # Rank correlations on peak
        rank_corr = compute_rank_correlation(
            ref_sub["peak"].values, exp_sub["peak"].values
        )

        # Top-k overlap
        top_k = compute_top_k_overlap(
            ref_sub["scenario_id"].values, ref_sub["peak"].values,
            exp_sub["scenario_id"].values, exp_sub["peak"].values,
            TOP_K_VALUES,
        )

        # Component-wise relative error
        comp_err = compute_component_relative_error(
            ref_sub, exp_sub, RANK_COMPONENTS
        )

        # Bootstrap CI
        boot = bootstrap_ci(
            ref_sub["peak"].values, exp_sub["peak"].values
        )

        all_results[exp_name] = {
            "description": exp["desc"],
            "n_scenarios": int(len(common)),
            "rank_correlation": rank_corr,
            "top_k_overlap": top_k,
            "component_relative_error": comp_err,
            "bootstrap_ci": boot,
        }

        score_matrix_list.append(exp_sub["peak"].values)
        exp_names.append(exp_name)

        sp = rank_corr.get("spearman", np.nan)
        kt = rank_corr.get("kendall", np.nan)
        print(f"  Spearman={sp:.4f}, Kendall={kt:.4f}, "
              f"n={len(common)}")

    # ICC across all experiments
    if len(score_matrix_list) >= 2:
        min_len = min(len(s) for s in score_matrix_list)
        mat = np.column_stack([s[:min_len] for s in score_matrix_list])
        icc_val = compute_icc(mat)
        all_results["icc"] = icc_val
        print(f"\n  ICC across all conditions: {icc_val}")

    # Save outputs
    # temporal_robustness.csv
    rob_rows = []
    for exp_name, res in all_results.items():
        if exp_name == "icc":
            continue
        row = {
            "experiment": exp_name,
            "description": res["description"],
            "n_scenarios": res["n_scenarios"],
            "spearman": res["rank_correlation"].get("spearman"),
            "kendall": res["rank_correlation"].get("kendall"),
            "boot_ci_lower": res["bootstrap_ci"].get("ci_lower"),
            "boot_ci_upper": res["bootstrap_ci"].get("ci_upper"),
        }
        for k, v in res.get("top_k_overlap", {}).items():
            row[k] = v
        rob_rows.append(row)

    df_robustness = pd.DataFrame(rob_rows)
    rob_path = os.path.join(output_root, "temporal_robustness.csv")
    df_robustness.to_csv(rob_path, index=False)
    print(f"\n[OK] {rob_path}")

    # ranking_stability.csv
    stab_rows = []
    for exp_name, res in all_results.items():
        if exp_name == "icc":
            continue
        for comp, err in res.get("component_relative_error", {}).items():
            stab_rows.append({
                "experiment": exp_name,
                "component": comp,
                **err,
            })
    df_stability = pd.DataFrame(stab_rows)
    stab_path = os.path.join(output_root, "ranking_stability.csv")
    df_stability.to_csv(stab_path, index=False)
    print(f"[OK] {stab_path}")

    # Full report
    report_path = os.path.join(output_root, "robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"[OK] {report_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Temporal Robustness Analysis")
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    pred_path = os.path.join(args.output_root, "frame_predictions.parquet")
    if not os.path.isfile(pred_path):
        print(f"[FATAL] {pred_path} not found.")
        sys.exit(1)

    df_pred = pd.read_parquet(pred_path)
    if args.max_scenarios:
        keep = df_pred["scenario_id"].unique()[:args.max_scenarios]
        df_pred = df_pred[df_pred["scenario_id"].isin(keep)]

    print(f"[Load] {len(df_pred)} frames, "
          f"{df_pred['scenario_id'].nunique()} scenarios")

    run_robustness_analysis(df_pred, args.output_root)


if __name__ == "__main__":
    main()
