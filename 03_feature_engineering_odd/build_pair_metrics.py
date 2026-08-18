#!/usr/bin/env python3
"""
DISABLED — Legacy Circle-Based Pair Metrics Builder
=====================================================
This module uses fast_ttc (numerical stepping Circle TTC), not the
canonical OBB swept SAT analytical method (obb_ttc_swept).

CANONICAL PATH:
  Pair metrics are generated inline within compute_obb_matched_pairs.py
  using compute_ttc_obb_swept() per pair.
"""
raise RuntimeError(
    "DISABLED: build_pair_metrics.py uses fast_ttc (numerical stepping Circle TTC), "
    "not the canonical OBB swept SAT method.\n"
    "Pair metrics are generated inline by: phase2_womd/compute_obb_matched_pairs.py"
)

# ── Original code below (never reached) ──
"""
Pairwise SDC↔Track Metrics Builder
=====================================
For each timestep, pairs the SDC track with every other eligible dynamic
track and computes OBB-based TTC metrics.

Eligible pair criteria:
  - object_type ∈ {TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST}
  - Both SDC and target have valid=True at this timestep
  - Euclidean distance ≤ PAIR_RADIUS_M
  - Target has valid geometry (length > 0, width > 0)
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from phase2_womd.fast_ttc import (
    DT_STEP,
    T_MAX_S,
    AgentBox,
    TTCResult,
    compute_ttc_fast as compute_ttc_obb,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAIR_RADIUS_M = 70.0
ELIGIBLE_TYPES = {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}


def build_pair_metrics(
    df_agent: pd.DataFrame,
    scenario_id: str,
    sdc_track_id: int,
    timestamps: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build pairwise TTC metrics for one scenario.

    Args:
        df_agent: agent_state DataFrame filtered to one scenario
        scenario_id: scenario identifier
        sdc_track_id: SDC track_id
        timestamps: optional series of timestamp_seconds

    Returns:
        DataFrame with one row per (time_index, target_track_id) pair + derived_* columns.
    """
    df_sc = df_agent[df_agent["scenario_id"] == scenario_id].copy()
    if df_sc.empty:
        return _empty_pair_df()

    time_indices = sorted(df_sc["time_index"].unique())
    pair_rows: List[Dict] = []

    for t_idx in time_indices:
        df_t = df_sc[df_sc["time_index"] == t_idx]

        # Get SDC state
        sdc_rows = df_t[df_t["track_id"] == sdc_track_id]
        if sdc_rows.empty:
            continue

        sdc = sdc_rows.iloc[0]
        if not sdc["valid"]:
            continue  # invalid ego — handled at frame level

        ego_box = _row_to_box(sdc)
        if ego_box is None:
            continue

        ts_val = sdc.get("timestamp_seconds", np.nan)

        # Find eligible targets
        targets = df_t[
            (df_t["track_id"] != sdc_track_id)
            & (df_t["valid"] == True)
            & (df_t["object_type"].isin(ELIGIBLE_TYPES))
        ]

        for _, tgt in targets.iterrows():
            tgt_box = _row_to_box(tgt)
            if tgt_box is None:
                continue

            # Distance filter
            dist = math.sqrt(
                (ego_box.cx - tgt_box.cx) ** 2
                + (ego_box.cy - tgt_box.cy) ** 2
            )
            if dist > PAIR_RADIUS_M:
                continue

            # Compute TTC
            result = compute_ttc_obb(ego_box, tgt_box)

            pair_rows.append({
                "scenario_id": scenario_id,
                "time_index": t_idx,
                "timestamp_seconds": ts_val,
                "ego_track_id": sdc_track_id,
                "target_track_id": tgt["track_id"],
                "target_object_type": tgt["object_type"],
                "pair_distance_m": dist,
                "derived_ttc_2d_s": result.derived_ttc_2d_s,
                "derived_dtc_m": result.derived_dtc_m,
                "derived_closing_speed_mps": result.derived_closing_speed_mps,
                "derived_overlap_now": result.derived_overlap_now,
                "derived_hit_future": result.derived_hit_future,
                "derived_pair_valid": result.derived_pair_valid,
                "derived_ttc_status": result.derived_ttc_status,
            })

    if not pair_rows:
        return _empty_pair_df()

    return pd.DataFrame(pair_rows)


def build_all_pair_metrics(
    df_agent: pd.DataFrame,
    df_scenario: pd.DataFrame,
) -> pd.DataFrame:
    """Build pair metrics for all scenarios.

    Args:
        df_agent: full agent_state DataFrame
        df_scenario: scenario_table DataFrame (needs scenario_id, sdc_track_index)

    Returns:
        Combined pair metrics DataFrame across all scenarios.
    """
    all_dfs = []

    for _, sc_row in df_scenario.iterrows():
        sid = sc_row["scenario_id"]
        # sdc_track_index is the index into tracks[], but we need the track_id
        # We get the SDC track_id from agent_state where is_sdc=True
        sdc_rows = df_agent[
            (df_agent["scenario_id"] == sid) & (df_agent["is_sdc"] == True)
        ]
        if sdc_rows.empty:
            continue
        sdc_track_id = sdc_rows.iloc[0]["track_id"]

        pair_df = build_pair_metrics(df_agent, sid, sdc_track_id)
        if not pair_df.empty:
            all_dfs.append(pair_df)

    if not all_dfs:
        return _empty_pair_df()

    return pd.concat(all_dfs, ignore_index=True)


def _row_to_box(row: pd.Series) -> Optional[AgentBox]:
    """Convert a DataFrame row to an AgentBox. Returns None if geometry invalid."""
    length = row.get("length", 0)
    width = row.get("width", 0)

    if (not np.isfinite(length) or not np.isfinite(width)
            or length <= 0 or width <= 0):
        return None

    cx = row.get("center_x", np.nan)
    cy = row.get("center_y", np.nan)
    heading = row.get("heading", 0.0)
    vx = row.get("velocity_x", 0.0)
    vy = row.get("velocity_y", 0.0)

    if not np.isfinite(cx) or not np.isfinite(cy):
        return None

    # Replace NaN velocity with 0 (valid state but possibly missing velocity)
    vx = float(vx) if np.isfinite(vx) else 0.0
    vy = float(vy) if np.isfinite(vy) else 0.0
    heading = float(heading) if np.isfinite(heading) else 0.0

    return AgentBox(
        cx=float(cx),
        cy=float(cy),
        length=float(length),
        width=float(width),
        heading=heading,
        vx=vx,
        vy=vy,
        valid=bool(row.get("valid", True)),
    )


def _empty_pair_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns."""
    return pd.DataFrame(columns=[
        "scenario_id", "time_index", "timestamp_seconds",
        "ego_track_id", "target_track_id", "target_object_type",
        "pair_distance_m",
        "derived_ttc_2d_s", "derived_dtc_m", "derived_closing_speed_mps",
        "derived_overlap_now", "derived_hit_future",
        "derived_pair_valid", "derived_ttc_status",
    ])
