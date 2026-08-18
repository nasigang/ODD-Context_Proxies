#!/usr/bin/env python3
"""
Scenario-Level Diagnostics
============================
Computes two frame-level curves and a 6-component diagnostic vector
from no-warp model frame predictions.

Frame curves:
  R_t^τ  — context-conditioned TTC lower-tail risk probability
           R_t = p_e · Φ((log(τ) − μ) / σ)

  S_t    — observed-event conditional lower-tail surprise
           S = −log(max(F_θ(y_t | C, Z), ε))
           Only computed when exposure=1 AND target_status='event'

Diagnostic vector D_s^risk (6 components):
  1. peak                  — max(R_t) over scenario
  2. exceedance_duration   — Σ(dt) where R_t > theta_on (timestamp-based)
  3. event_count           — hysteresis count with merge_gap + min_duration
  4. max_event_duration    — longest continuous exceedance
  5. recovery_time         — time from last exceedance exit to confirmed recovery
  6. predictive_uncertainty— q90−q10 predictive interval width

LSE is stored as auxiliary scalar only.

Usage:
    python phase2_womd/scenario_diagnostics.py --max-scenarios 20
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from phase2_womd.diag_config import DiagConfig
from phase2_womd.obb_ttc import T_MAX_S, T_MIN_S


# ---------------------------------------------------------------------------
# Frame curve computation
# ---------------------------------------------------------------------------

def compute_Rt(
    pred_p_exposure: np.ndarray,
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    tau: float = 3.0,
) -> np.ndarray:
    """Compute R_t^τ: context-conditioned lower-tail risk probability.

    R_t = p_e · Φ((log(τ) − μ) / σ)
    """
    log_tau = math.log(max(tau, T_MIN_S))
    sigma = np.maximum(pred_sigma, 1e-12)
    z = (log_tau - pred_mu) / sigma
    return pred_p_exposure * norm.cdf(z)


def compute_St(
    y_log_ttc: np.ndarray,
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    exposure_flag: np.ndarray,
    target_status: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute S_t: observed-event conditional surprise.

    S = −log(max(F_θ(y_t | μ, σ), ε))
    Only computed when exposure=1 AND target_status='event'.
    Other frames get NaN.
    """
    n = len(y_log_ttc)
    St = np.full(n, np.nan)

    for i in range(n):
        if exposure_flag[i] == 1 and target_status[i] == "event":
            sigma = max(pred_sigma[i], 1e-12)
            z = (y_log_ttc[i] - pred_mu[i]) / sigma
            F_val = norm.cdf(z)
            St[i] = -math.log(max(F_val, epsilon))

    return St


# ---------------------------------------------------------------------------
# Hysteresis event detection
# ---------------------------------------------------------------------------

def detect_exceedance_events(
    Rt: np.ndarray,
    timestamps: np.ndarray,
    cfg: DiagConfig,
) -> List[Dict]:
    """Detect exceedance events using hysteresis thresholds.

    Uses theta_on for entry, theta_off for exit.
    Applies merge_gap and minimum_event_duration filtering.

    Returns list of event dicts with start_time, end_time, duration.
    """
    n = len(Rt)
    if n == 0:
        return []

    # Phase 1: raw exceedance intervals using hysteresis
    raw_events = []
    in_event = False
    event_start = None

    for i in range(n):
        if not in_event:
            if Rt[i] >= cfg.theta_on:
                in_event = True
                event_start = timestamps[i]
        else:
            if Rt[i] < cfg.theta_off:
                in_event = False
                event_end = timestamps[i]
                raw_events.append({
                    "start_time": event_start,
                    "end_time": event_end,
                    "duration": event_end - event_start,
                })

    # Close open event at scenario end
    if in_event and event_start is not None:
        raw_events.append({
            "start_time": event_start,
            "end_time": timestamps[-1],
            "duration": timestamps[-1] - event_start,
        })

    if not raw_events:
        return []

    # Phase 2: merge events separated by ≤ merge_gap
    merged = [raw_events[0].copy()]
    for ev in raw_events[1:]:
        gap = ev["start_time"] - merged[-1]["end_time"]
        if gap <= cfg.merge_gap_s:
            merged[-1]["end_time"] = ev["end_time"]
            merged[-1]["duration"] = (
                merged[-1]["end_time"] - merged[-1]["start_time"]
            )
        else:
            merged.append(ev.copy())

    # Phase 3: filter by minimum duration
    filtered = [
        ev for ev in merged
        if ev["duration"] >= cfg.minimum_event_duration_s
    ]

    return filtered


# ---------------------------------------------------------------------------
# Recovery detection
# ---------------------------------------------------------------------------

def compute_recovery(
    Rt: np.ndarray,
    timestamps: np.ndarray,
    events: List[Dict],
    cfg: DiagConfig,
) -> Tuple[float, bool]:
    """Compute recovery time after last exceedance event.

    Recovery is confirmed when R_t stays below theta_off for
    recovery_hold_time_s continuously.

    Returns:
        (recovery_time, recovery_censored)
        recovery_time: time from last event end to confirmed recovery
        recovery_censored: True if scenario ended before recovery
    """
    if not events:
        return 0.0, False

    last_event_end = events[-1]["end_time"]
    scenario_end = timestamps[-1]

    # Find frames after last event
    post_event_mask = timestamps > last_event_end
    if not post_event_mask.any():
        return 0.0, True  # scenario ended during/at event

    # Track continuous time below theta_off
    hold_start = None
    for i in range(len(timestamps)):
        if timestamps[i] <= last_event_end:
            continue

        if Rt[i] < cfg.theta_off:
            if hold_start is None:
                hold_start = timestamps[i]
            hold_duration = timestamps[i] - hold_start
            if hold_duration >= cfg.recovery_hold_time_s:
                recovery_time = timestamps[i] - last_event_end
                return recovery_time, False
        else:
            hold_start = None  # reset

    # Scenario ended without confirmed recovery
    return scenario_end - last_event_end, True


# ---------------------------------------------------------------------------
# Diagnostic vector
# ---------------------------------------------------------------------------

def compute_diagnostic_vector(
    Rt: np.ndarray,
    St: np.ndarray,
    timestamps: np.ndarray,
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    cfg: DiagConfig,
) -> Dict:
    """Compute 6-component diagnostic vector D_s^risk for one scenario.

    Components:
      1. peak — max(R_t)
      2. exceedance_duration — timestamp-based integral above theta_on
      3. event_count — hysteresis with merge + min_duration
      4. max_event_duration — longest event
      5. recovery_time — time to confirmed recovery after last event
      6. predictive_uncertainty — q90−q10 predictive interval width
    """
    n = len(Rt)
    if n == 0:
        return _empty_vector()

    # 1. Peak
    peak = float(np.nanmax(Rt))

    # 2. Exceedance duration (timestamp-based integration)
    exceedance_duration = 0.0
    for i in range(1, n):
        if Rt[i] >= cfg.theta_on:
            dt = timestamps[i] - timestamps[i - 1]
            exceedance_duration += dt

    # 3+4. Event count and max event duration
    events = detect_exceedance_events(Rt, timestamps, cfg)
    event_count = len(events)
    max_event_duration = (
        max(ev["duration"] for ev in events) if events else 0.0
    )

    # 5. Recovery time
    recovery_time, recovery_censored = compute_recovery(
        Rt, timestamps, events, cfg
    )

    # 6. Predictive uncertainty: q90−q10 interval width
    # For Gaussian: width = σ · (z_0.9 − z_0.1) = σ · 2.5631
    z90 = norm.ppf(0.90)
    z10 = norm.ppf(0.10)
    interval_widths = pred_sigma * (z90 - z10)
    predictive_uncertainty = float(np.nanmean(interval_widths))

    # Auxiliary: LSE (log-sum-exp of R_t) — stored as secondary
    Rt_safe = np.where(Rt > 0, Rt, 1e-30)
    lse_score = float(np.log(np.sum(np.exp(np.log(Rt_safe)))))

    # Auxiliary: mean R_t
    mean_Rt = float(np.nanmean(Rt))

    return {
        # Primary 6-component vector
        "peak": round(peak, 6),
        "exceedance_duration": round(exceedance_duration, 4),
        "event_count": event_count,
        "max_event_duration": round(max_event_duration, 4),
        "recovery_time": round(recovery_time, 4),
        "predictive_uncertainty": round(predictive_uncertainty, 6),
        # Auxiliary (NOT primary results)
        "recovery_censored": recovery_censored,
        "lse_score": round(lse_score, 4),
        "mean_Rt": round(mean_Rt, 6),
    }


def _empty_vector() -> Dict:
    return {
        "peak": 0.0,
        "exceedance_duration": 0.0,
        "event_count": 0,
        "max_event_duration": 0.0,
        "recovery_time": 0.0,
        "predictive_uncertainty": 0.0,
        "recovery_censored": False,
        "lse_score": 0.0,
        "mean_Rt": 0.0,
    }


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all_diagnostics(
    df_pred: pd.DataFrame,
    cfg: Optional[DiagConfig] = None,
    tau: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute diagnostics for all scenarios in frame predictions.

    Args:
        df_pred: frame_predictions DataFrame with columns:
            scenario_id, time_index, timestamp_seconds (optional),
            pred_p_exposure, pred_mu, pred_sigma,
            y_log_ttc, ttc_censored, exposure_flag, target_status
        cfg: DiagConfig instance
        tau: override τ for R_t computation

    Returns:
        (scenario_vectors, frame_curves)
        scenario_vectors: one row per scenario with D_s^risk
        frame_curves: one row per frame with R_t, S_t
    """
    if cfg is None:
        cfg = DiagConfig()
    if tau is None:
        tau = cfg.tau_primary

    scenarios = df_pred["scenario_id"].unique()
    vec_rows = []
    curve_rows = []

    for sid in scenarios:
        df_sc = df_pred[df_pred["scenario_id"] == sid].sort_values("time_index")

        # Use actual timestamps, fallback to time_index * 0.1 if missing
        if "timestamp_seconds" in df_sc.columns and df_sc["timestamp_seconds"].notna().any():
            ts = df_sc["timestamp_seconds"].values.astype(float)
            # Normalize to start at 0
            ts = ts - ts[0]
        else:
            ts = df_sc["time_index"].values.astype(float) * 0.1

        p_e = df_sc["pred_p_exposure"].values.astype(float)
        mu = df_sc["pred_mu"].values.astype(float)
        sigma = df_sc["pred_sigma"].values.astype(float)
        y_log = df_sc["y_log_ttc"].values.astype(float)
        exp_flag = df_sc["exposure_flag"].values
        status = df_sc["target_status"].values

        # Frame curves
        Rt = compute_Rt(p_e, mu, sigma, tau=tau)
        St = compute_St(y_log, mu, sigma, exp_flag, status, epsilon=cfg.epsilon)

        # Diagnostic vector
        dvec = compute_diagnostic_vector(Rt, St, ts, mu, sigma, cfg)
        dvec["scenario_id"] = sid
        vec_rows.append(dvec)

        # Store frame curves
        for i in range(len(df_sc)):
            curve_rows.append({
                "scenario_id": sid,
                "time_index": int(df_sc.iloc[i]["time_index"]),
                "timestamp_rel": float(ts[i]),
                "Rt": float(Rt[i]),
                "St": float(St[i]) if np.isfinite(St[i]) else None,
            })

    df_vectors = pd.DataFrame(vec_rows)
    df_curves = pd.DataFrame(curve_rows)
    return df_vectors, df_curves


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Scenario Diagnostics")
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    # Load frame predictions
    pred_path = os.path.join(args.output_root, "frame_predictions.parquet")
    if not os.path.isfile(pred_path):
        print(f"[FATAL] {pred_path} not found. Run train_phase2_models.py first.")
        sys.exit(1)

    df_pred = pd.read_parquet(pred_path)
    if args.max_scenarios:
        keep = df_pred["scenario_id"].unique()[:args.max_scenarios]
        df_pred = df_pred[df_pred["scenario_id"].isin(keep)]

    print(f"[Load] {len(df_pred)} frames, "
          f"{df_pred['scenario_id'].nunique()} scenarios")

    cfg = DiagConfig()
    df_vectors, df_curves = compute_all_diagnostics(df_pred, cfg)

    # Save outputs
    vec_path = os.path.join(args.output_root, "scenario_diagnostic_vector.parquet")
    df_vectors.to_parquet(vec_path, index=False)
    print(f"[OK] Diagnostic vectors: {vec_path} ({len(df_vectors)} scenarios)")

    fig_path = os.path.join(args.output_root, "diagnostic_figure_data.parquet")
    df_curves.to_parquet(fig_path, index=False)
    print(f"[OK] Frame curves: {fig_path} ({len(df_curves)} frames)")

    # Print summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC VECTOR SUMMARY")
    print(f"{'='*60}")
    for col in ["peak", "exceedance_duration", "event_count",
                 "max_event_duration", "recovery_time",
                 "predictive_uncertainty"]:
        if col in df_vectors.columns:
            vals = df_vectors[col].dropna()
            print(f"  {col:25s}  mean={vals.mean():.4f}  "
                  f"std={vals.std():.4f}  "
                  f"max={vals.max():.4f}")
    n_censored = df_vectors["recovery_censored"].sum()
    print(f"  recovery_censored: {n_censored}/{len(df_vectors)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
