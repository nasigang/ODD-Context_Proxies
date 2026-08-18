#!/usr/bin/env python3
"""
DISABLED — Legacy Circle-Based Frame Target Builder
=====================================================
This module builds frame targets from Circle TTC (derived_ttc_2d_s from fast_ttc),
not from the canonical OBB swept SAT method. It also lacks any-pair overlap
precedence (checks only the idxmin pair).

CANONICAL PATH:
  Frame targets are generated inline within compute_obb_matched_pairs.py
  using process_scenario_obb() with correct overlap-first precedence.
"""
raise RuntimeError(
    "DISABLED: build_frame_targets.py uses Circle TTC (derived_ttc_2d_s) "
    "from the legacy fast_ttc module, not the canonical OBB swept SAT method.\n"
    "Frame targets are generated inline by: phase2_womd/compute_obb_matched_pairs.py"
)

# ── Original code below (never reached) ──
"""
Frame-Level Target Builder
============================
Aggregates pairwise TTC metrics into per-frame targets for survival
analysis / risk prediction.

Every frame is preserved — frames with no eligible pairs get
target_status="no_exposure", distinct from "right_censored".

Target status values (mutually exclusive):
  - event           : E_t=1, TTC ≤ T_MAX_S, overlap found
  - right_censored  : E_t=1, no overlap within T_MAX_S
  - no_exposure     : no eligible pairs at this frame
  - invalid_ego_state : SDC state not valid at this frame
  - invalid_frame   : frame has no valid agents at all

y_log_ttc:
  - event           : log(max(TTC, T_MIN_S))
  - right_censored  : log(T_MAX_S)
  - no_exposure     : NaN
  - invalid_*       : NaN
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from phase2_womd.obb_ttc import T_MAX_S, T_MIN_S


def build_frame_targets(
    df_agent: pd.DataFrame,
    df_pairs: pd.DataFrame,
    scenario_id: str,
    sdc_track_id: int,
) -> pd.DataFrame:
    """Build frame-level targets for one scenario.

    Args:
        df_agent: agent_state DataFrame for this scenario
        df_pairs: pair_metrics DataFrame for this scenario (may be empty)
        scenario_id: scenario identifier
        sdc_track_id: SDC track_id

    Returns:
        DataFrame with one row per frame (time_index), all frames preserved.
    """
    df_sc = df_agent[df_agent["scenario_id"] == scenario_id]
    if df_sc.empty:
        return _empty_targets_df()

    time_indices = sorted(df_sc["time_index"].unique())
    rows: List[Dict] = []

    for t_idx in time_indices:
        df_t = df_sc[df_sc["time_index"] == t_idx]

        # Get timestamp
        ts_val = df_t["timestamp_seconds"].iloc[0] if not df_t.empty else np.nan

        # Check if any valid agents exist
        n_valid = df_t["valid"].sum()
        if n_valid == 0:
            rows.append(_make_row(
                scenario_id, t_idx, ts_val,
                status="invalid_frame",
            ))
            continue

        # Check SDC validity
        sdc_rows = df_t[df_t["track_id"] == sdc_track_id]
        if sdc_rows.empty or not sdc_rows.iloc[0]["valid"]:
            rows.append(_make_row(
                scenario_id, t_idx, ts_val,
                status="invalid_ego_state",
            ))
            continue

        # Get pairs for this frame
        frame_pairs = pd.DataFrame()
        if not df_pairs.empty:
            frame_pairs = df_pairs[
                (df_pairs["scenario_id"] == scenario_id)
                & (df_pairs["time_index"] == t_idx)
                & (df_pairs["derived_pair_valid"] == True)
            ]

        if frame_pairs.empty:
            rows.append(_make_row(
                scenario_id, t_idx, ts_val,
                status="no_exposure",
            ))
            continue

        # We have eligible pairs — find minimum TTC
        n_eligible = len(frame_pairs)
        idx_min = frame_pairs["derived_ttc_2d_s"].idxmin()
        best = frame_pairs.loc[idx_min]

        ttc_val = best["derived_ttc_2d_s"]
        ttc_status = best["derived_ttc_status"]

        # Determine target_status
        if ttc_status == "event":
            target_status = "event"
            ttc_censored = False
            y_log = math.log(max(ttc_val, T_MIN_S))
        elif ttc_status == "right_censored":
            target_status = "right_censored"
            ttc_censored = True
            y_log = math.log(T_MAX_S)
        else:
            target_status = "right_censored"
            ttc_censored = True
            y_log = math.log(T_MAX_S)

        rows.append({
            "scenario_id": scenario_id,
            "time_index": t_idx,
            "timestamp_seconds": ts_val,
            "exposure_flag": 1,
            "n_eligible_pairs": n_eligible,
            "ttc_min_s": ttc_val,
            "ttc_censored": ttc_censored,
            "y_log_ttc": y_log,
            "target_object_id": int(best["target_track_id"]),
            "target_object_type": best["target_object_type"],
            "target_status": target_status,
        })

    if not rows:
        return _empty_targets_df()

    return pd.DataFrame(rows)


def build_all_frame_targets(
    df_agent: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_scenario: pd.DataFrame,
) -> pd.DataFrame:
    """Build frame targets for all scenarios.

    Args:
        df_agent: full agent_state DataFrame
        df_pairs: full pair_metrics DataFrame
        df_scenario: scenario_table DataFrame

    Returns:
        Combined frame targets DataFrame across all scenarios.
    """
    all_dfs = []

    for _, sc_row in df_scenario.iterrows():
        sid = sc_row["scenario_id"]

        # Get SDC track_id
        sdc_rows = df_agent[
            (df_agent["scenario_id"] == sid) & (df_agent["is_sdc"] == True)
        ]
        if sdc_rows.empty:
            continue
        sdc_track_id = sdc_rows.iloc[0]["track_id"]

        ft = build_frame_targets(df_agent, df_pairs, sid, sdc_track_id)
        if not ft.empty:
            all_dfs.append(ft)

    if not all_dfs:
        return _empty_targets_df()

    return pd.concat(all_dfs, ignore_index=True)


def _make_row(
    scenario_id: str,
    time_index: int,
    timestamp_seconds: float,
    status: str,
) -> Dict:
    """Create a frame target row for non-exposure / invalid frames."""
    return {
        "scenario_id": scenario_id,
        "time_index": time_index,
        "timestamp_seconds": timestamp_seconds,
        "exposure_flag": 0,
        "n_eligible_pairs": 0,
        "ttc_min_s": np.nan,
        "ttc_censored": False,
        "y_log_ttc": np.nan,
        "target_object_id": None,
        "target_object_type": None,
        "target_status": status,
    }


def _empty_targets_df() -> pd.DataFrame:
    """Return an empty DataFrame with correct columns."""
    return pd.DataFrame(columns=[
        "scenario_id", "time_index", "timestamp_seconds",
        "exposure_flag", "n_eligible_pairs",
        "ttc_min_s", "ttc_censored", "y_log_ttc",
        "target_object_id", "target_object_type", "target_status",
    ])
