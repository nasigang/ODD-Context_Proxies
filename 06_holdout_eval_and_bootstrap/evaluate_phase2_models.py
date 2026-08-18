#!/usr/bin/env python3
"""
DISABLED — Legacy Phase 2 Model Evaluator
==========================================
This module auto-trains models without gate validation and evaluates
internal_holdout without freeze verification. Both behaviors violate
the production R2 contract.

CANONICAL PATH:
  Training:   phase2_womd/train_r2_models.py (requires gate attestation)
  Evaluation: phase2_womd/evaluate_frozen_r2.py (split allowlist enforced)
  Holdout:    phase2_womd/open_holdout_once.py (sentinel-guarded one-shot)
"""
raise RuntimeError(
    "DISABLED: evaluate_phase2_models.py is a legacy evaluator that auto-trains "
    "without gate validation and evaluates holdout without freeze verification.\n"
    "Use the canonical production path instead:\n"
    "  Training:   python phase2_womd/train_r2_models.py\n"
    "  Evaluation: phase2_womd.evaluate_frozen_r2.FrozenEvaluator\n"
    "  Holdout:    phase2_womd.open_holdout_once.open_holdout_once()"
)

# ── Original code below (never reached) ──
"""
Phase 2 Model Evaluation
==========================
Evaluates trained models with comprehensive metrics:
  - Censored NLL
  - Exposure gate Brier score
  - P(TTC ≤ τ) calibration at multiple thresholds
  - Censored / uncensored PIT histograms
  - Lower-tail quantile coverage
  - ODD-stratified calibration

Outputs:
  model_config.json       (from training)
  model_metrics.csv       (detailed per-split metrics)
  frame_predictions.parquet  (from training)
  leakage_audit.json      (from build_model_table)
  calibration_report.json (calibration + PIT + coverage)

Usage:
    python phase2_womd/evaluate_phase2_models.py --max-scenarios 20
"""

import argparse
import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from phase2_womd.build_model_table import (
    C_ODD_FEATURES,
    MODEL_FEATURES,
    Z_STATE_FEATURES,
)
from phase2_womd.obb_ttc import T_MAX_S, T_MIN_S
from phase2_womd.train_phase2_models import (
    MODEL_SPECS,
    HurdleModel,
    censored_nll,
    train_all_models,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

TAU_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50]


def compute_calibration(
    df: pd.DataFrame,
    pred: pd.DataFrame,
    model: HurdleModel,
) -> Dict:
    """Compute P(TTC ≤ τ) calibration at multiple thresholds."""
    results = {}

    exposed = df["exposure_flag"] == 1
    if exposed.sum() == 0:
        return {"note": "no exposed frames"}

    y_log = df.loc[exposed, "y_log_ttc"].values
    is_event = (df.loc[exposed, "target_status"] == "event").values
    p_e = pred.loc[exposed.values, "pred_p_exposure"].values
    mu = pred.loc[exposed.values, "pred_mu"].values
    sigma = pred.loc[exposed.values, "pred_sigma"].values

    for tau in TAU_THRESHOLDS:
        log_tau = math.log(max(tau, T_MIN_S))

        # Predicted P(TTC ≤ τ) = p_e · Φ((log(τ) - μ) / σ)
        z = (log_tau - mu) / np.maximum(sigma, 1e-12)
        pred_prob = p_e * norm.cdf(z)

        # Observed frequency: event with TTC ≤ τ
        obs = is_event & (y_log <= log_tau + 1e-9)
        obs_rate = float(obs.mean()) if len(obs) > 0 else 0.0
        pred_rate = float(pred_prob.mean()) if len(pred_prob) > 0 else 0.0

        results[f"tau_{tau}s"] = {
            "observed_rate": round(obs_rate, 6),
            "predicted_rate": round(pred_rate, 6),
            "absolute_error": round(abs(obs_rate - pred_rate), 6),
            "n_samples": int(exposed.sum()),
        }

    return results


def compute_pit(
    df: pd.DataFrame,
    pred: pd.DataFrame,
    model: HurdleModel,
) -> Dict:
    """Compute Probability Integral Transform for uncensored observations."""
    exposed = df["exposure_flag"] == 1
    events = df["target_status"] == "event"
    censored = df["target_status"] == "right_censored"

    result = {"uncensored_pit": {}, "censored_pit": {}}

    # Uncensored PIT: F(y | μ, σ) should be uniform
    event_mask = exposed & events
    if event_mask.sum() > 0:
        y = df.loc[event_mask, "y_log_ttc"].values
        mu = pred.loc[event_mask.values, "pred_mu"].values
        sigma = pred.loc[event_mask.values, "pred_sigma"].values
        z = (y - mu) / np.maximum(sigma, 1e-12)
        pit_vals = norm.cdf(z)

        # Histogram into 10 bins
        hist, _ = np.histogram(pit_vals, bins=10, range=(0, 1))
        result["uncensored_pit"] = {
            "histogram": hist.tolist(),
            "n_samples": int(event_mask.sum()),
            "mean": float(np.mean(pit_vals)),
            "std": float(np.std(pit_vals)),
        }

    # Censored PIT: S(y | μ, σ) should be well-behaved
    cens_mask = exposed & censored
    if cens_mask.sum() > 0:
        y = df.loc[cens_mask, "y_log_ttc"].values
        mu = pred.loc[cens_mask.values, "pred_mu"].values
        sigma = pred.loc[cens_mask.values, "pred_sigma"].values
        z = (y - mu) / np.maximum(sigma, 1e-12)
        surv_vals = norm.sf(z)  # survival function

        hist, _ = np.histogram(surv_vals, bins=10, range=(0, 1))
        result["censored_pit"] = {
            "histogram": hist.tolist(),
            "n_samples": int(cens_mask.sum()),
            "mean": float(np.mean(surv_vals)),
        }

    return result


def compute_quantile_coverage(
    df: pd.DataFrame,
    pred: pd.DataFrame,
    model: HurdleModel,
) -> Dict:
    """Compute lower-tail quantile coverage."""
    events = (df["target_status"] == "event") & (df["exposure_flag"] == 1)
    if events.sum() == 0:
        return {"note": "no events"}

    y = df.loc[events, "y_log_ttc"].values
    mu = pred.loc[events.values, "pred_mu"].values
    sigma = pred.loc[events.values, "pred_sigma"].values

    result = {}
    for q in QUANTILE_LEVELS:
        # Predicted q-th quantile: μ + σ · Φ^{-1}(q)
        z_q = norm.ppf(q)
        q_pred = mu + sigma * z_q

        # Coverage: fraction of events below predicted quantile
        coverage = float((y <= q_pred).mean())
        result[f"q_{q:.2f}"] = {
            "nominal": q,
            "actual_coverage": round(coverage, 4),
            "gap": round(coverage - q, 4),
        }

    return result


def compute_odd_stratified_calibration(
    df: pd.DataFrame,
    pred: pd.DataFrame,
) -> Dict:
    """Compute calibration stratified by ODD features."""
    result = {}

    # Stratify by traffic density (n_valid_agents)
    if "odd_n_valid_agents" in df.columns:
        agent_col = df["odd_n_valid_agents"].fillna(0)
        median_agents = agent_col.median()

        for label, mask in [
            ("low_traffic", agent_col <= median_agents),
            ("high_traffic", agent_col > median_agents),
        ]:
            subset = df[mask]
            sub_pred = pred[mask.values]
            if len(subset) == 0:
                continue

            exposed = subset["exposure_flag"] == 1
            if exposed.sum() == 0:
                result[label] = {"n_frames": len(subset), "brier": None}
                continue

            p_e = sub_pred["pred_p_exposure"].values
            y_e = subset["exposure_flag"].values.astype(float)
            valid = np.isfinite(p_e) & np.isfinite(y_e)
            brier = float(np.mean((p_e[valid] - y_e[valid]) ** 2))

            result[label] = {
                "n_frames": int(len(subset)),
                "brier": round(brier, 6),
                "exposure_rate": round(float(y_e[valid].mean()), 4),
            }

    # Stratify by signal presence
    if "odd_has_signal" in df.columns:
        for label, val in [("no_signal", 0), ("has_signal", 1)]:
            mask = df["odd_has_signal"] == val
            subset = df[mask]
            sub_pred = pred[mask.values]
            if len(subset) == 0:
                continue

            p_e = sub_pred["pred_p_exposure"].values
            y_e = subset["exposure_flag"].values.astype(float)
            valid = np.isfinite(p_e) & np.isfinite(y_e)
            if valid.sum() == 0:
                continue
            brier = float(np.mean((p_e[valid] - y_e[valid]) ** 2))

            result[label] = {
                "n_frames": int(len(subset)),
                "brier": round(brier, 6),
                "exposure_rate": round(float(y_e[valid].mean()), 4),
            }

    return result


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_all_models(
    output_root: str,
    max_scenarios: Optional[int] = None,
    womd_root: Optional[str] = None,
) -> Dict:
    """Evaluate all trained models."""
    os.makedirs(output_root, exist_ok=True)

    # Ensure models are trained
    config_path = os.path.join(output_root, "model_config.json")
    if not os.path.isfile(config_path):
        print("[Train] Model config not found, training first...")
        train_all_models(
            output_root=output_root,
            max_scenarios=max_scenarios,
            womd_root=womd_root,
        )

    # Load model table and predictions
    table_path = os.path.join(output_root, "model_table.parquet")
    if not os.path.isfile(table_path):
        print("[FATAL] model_table.parquet not found")
        sys.exit(1)

    df = pd.read_parquet(table_path)

    with open(config_path) as f:
        model_configs = json.load(f)

    # Check leakage audit
    audit_path = os.path.join(output_root, "leakage_audit.json")
    if os.path.isfile(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)
        if audit.get("status") == "FAIL":
            print("[FATAL] Leakage audit FAILED — cannot evaluate.")
            sys.exit(1)
        print(f"[OK] Leakage audit: {audit['status']}")

    # Evaluate each model
    print("\n" + "=" * 70)
    print("PHASE 2 MODEL EVALUATION")
    print("=" * 70)

    all_results = {}
    eval_splits = ["train", "internal_val", "internal_holdout"]

    for model_key, spec in MODEL_SPECS.items():
        config = model_configs.get(model_key, {})
        if config.get("status") == "placeholder":
            all_results[model_key] = {"status": "placeholder"}
            continue

        print(f"\n--- Evaluating {model_key}: {spec['name']} ---")

        # Reconstruct model
        model = HurdleModel(model_key)
        features = spec["features"]
        add_int = spec.get("interactions", False)

        df_train = df[df["split"] == "train"]
        if model_key == "M0":
            model.fit_unconditional(df_train)
        elif spec.get("separate_gate_expert"):
            model.fit_gate_expert(df_train, features)
        else:
            model.fit_conditional(df_train, features, add_interactions=add_int)

        model_results = {"status": "evaluated", "splits": {}}

        for split_name in eval_splits:
            df_split = df[df["split"] == split_name]
            if df_split.empty:
                continue

            pred = model.predict(df_split, features, add_interactions=add_int)

            # Metrics
            metrics = {}

            # Censored NLL
            exposed = df_split["exposure_flag"] == 1
            if exposed.sum() > 0:
                y = df_split.loc[exposed, "y_log_ttc"].values
                delta = (df_split.loc[exposed, "target_status"] == "event").values.astype(float)
                mu = pred.loc[exposed.values, "pred_mu"].values
                sigma = pred.loc[exposed.values, "pred_sigma"].values
                valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
                if valid.sum() > 0:
                    metrics["censored_nll"] = censored_nll(
                        y[valid], delta[valid], mu[valid], sigma[valid]
                    )

            # Brier
            p_e = pred["pred_p_exposure"].values
            y_e = df_split["exposure_flag"].values.astype(float)
            v = np.isfinite(p_e) & np.isfinite(y_e)
            if v.sum() > 0:
                metrics["gate_brier"] = float(np.mean((p_e[v] - y_e[v]) ** 2))

            # Calibration
            metrics["calibration"] = compute_calibration(df_split, pred, model)

            # PIT
            metrics["pit"] = compute_pit(df_split, pred, model)

            # Quantile coverage
            metrics["quantile_coverage"] = compute_quantile_coverage(
                df_split, pred, model
            )

            # ODD-stratified
            if split_name == "internal_val":
                metrics["odd_stratified"] = compute_odd_stratified_calibration(
                    df_split, pred
                )

            metrics["n_frames"] = int(len(df_split))
            metrics["n_scenarios"] = int(df_split["scenario_id"].nunique())

            model_results["splits"][split_name] = metrics

            nll = metrics.get("censored_nll", "N/A")
            brier = metrics.get("gate_brier", "N/A")
            nll_str = f"{nll:.4f}" if isinstance(nll, float) else str(nll)
            brier_str = f"{brier:.4f}" if isinstance(brier, float) else str(brier)
            print(f"  [{split_name}] NLL={nll_str}, Brier={brier_str}, "
                  f"frames={metrics['n_frames']}")

        all_results[model_key] = model_results

    # Save calibration report
    cal_path = os.path.join(output_root, "calibration_report.json")
    with open(cal_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] Calibration report: {cal_path}")

    # Update model metrics CSV with full results
    _save_full_metrics_csv(all_results, output_root)

    _print_summary(all_results)
    return all_results


def _save_full_metrics_csv(results: Dict, output_dir: str):
    rows = []
    for model_key, res in results.items():
        if res.get("status") == "placeholder":
            rows.append({"model": model_key, "split": "N/A", "status": "placeholder"})
            continue
        for split_name, metrics in res.get("splits", {}).items():
            row = {
                "model": model_key,
                "split": split_name,
                "censored_nll": metrics.get("censored_nll"),
                "gate_brier": metrics.get("gate_brier"),
                "n_frames": metrics.get("n_frames"),
                "n_scenarios": metrics.get("n_scenarios"),
            }
            # Add calibration deltas
            for tau_key, cal in metrics.get("calibration", {}).items():
                if isinstance(cal, dict):
                    row[f"cal_{tau_key}_err"] = cal.get("absolute_error")
            rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "model_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Updated metrics: {path}")


def _print_summary(results: Dict):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<8} {'Split':<18} {'NLL':>8} {'Brier':>8} {'Frames':>8}")
    print("-" * 52)
    for model_key, res in results.items():
        if res.get("status") == "placeholder":
            print(f"{model_key:<8} {'placeholder':<18}")
            continue
        for split_name, metrics in res.get("splits", {}).items():
            nll = metrics.get("censored_nll", "")
            brier = metrics.get("gate_brier", "")
            nll_s = f"{nll:.4f}" if isinstance(nll, float) else "N/A"
            brier_s = f"{brier:.4f}" if isinstance(brier, float) else "N/A"
            print(f"{model_key:<8} {split_name:<18} {nll_s:>8} {brier_s:>8} "
                  f"{metrics.get('n_frames', 0):>8}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2 Model Evaluation")
    ap.add_argument("--womd-root",
                    default=os.environ.get("WOMD_ROOT", "/mnt/womd"))
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    evaluate_all_models(
        output_root=args.output_root,
        max_scenarios=args.max_scenarios,
        womd_root=args.womd_root,
    )


if __name__ == "__main__":
    main()
