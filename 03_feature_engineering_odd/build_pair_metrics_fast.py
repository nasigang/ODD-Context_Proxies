#!/usr/bin/env python3
"""
Vectorized Pairwise TTC Builder
==================================
Computes all SDC↔target pair metrics using numpy vectorization.
No Python-level loops over individual pairs.

Uses analytical circle-based TTC (quadratic solve, O(1) per pair).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

PAIR_RADIUS_M = 70.0
T_MAX_S = 10.0
ELIGIBLE_TYPES = {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}


def build_pair_metrics_vectorized(
    df_agent_scenario: pd.DataFrame,
    scenario_id: str,
    sdc_track_id: int,
) -> pd.DataFrame:
    """Build pair metrics for one scenario using vectorized operations.

    Args:
        df_agent_scenario: agent_state DataFrame ALREADY FILTERED to this scenario
        scenario_id: scenario id
        sdc_track_id: SDC track_id

    Returns:
        DataFrame with pair metrics
    """
    df = df_agent_scenario

    if df.empty:
        return _empty_pair_df()

    time_indices = sorted(df["time_index"].unique())
    all_parts = []

    # Pre-extract SDC data as dict: {time_index: Series}
    sdc_mask = df["track_id"] == sdc_track_id
    df_sdc = df[sdc_mask]

    if df_sdc.empty:
        return _empty_pair_df()

    sdc_by_t = {row["time_index"]: row for _, row in df_sdc.iterrows()}

    # Pre-filter eligible targets
    tgt_mask = (
        (~sdc_mask) &
        (df["valid"] == True) &
        (df["object_type"].isin(ELIGIBLE_TYPES)) &
        (df["length"] > 0) &
        (df["width"] > 0)
    )
    df_tgt = df[tgt_mask]

    if df_tgt.empty:
        return _empty_pair_df()

    # Group targets by time_index
    tgt_grouped = {t: g for t, g in df_tgt.groupby("time_index")}

    for t_idx in time_indices:
        sdc_row = sdc_by_t.get(t_idx)
        if sdc_row is None:
            continue
        if not sdc_row["valid"]:
            continue

        ego_cx = float(sdc_row["center_x"])
        ego_cy = float(sdc_row["center_y"])
        if not (np.isfinite(ego_cx) and np.isfinite(ego_cy)):
            continue

        ego_vx = float(sdc_row.get("velocity_x", 0.0))
        ego_vy = float(sdc_row.get("velocity_y", 0.0))
        ego_len = float(sdc_row["length"])
        ego_wid = float(sdc_row["width"])
        if ego_len <= 0 or ego_wid <= 0:
            continue

        ego_vx = ego_vx if np.isfinite(ego_vx) else 0.0
        ego_vy = ego_vy if np.isfinite(ego_vy) else 0.0
        ego_R = 0.5 * math.sqrt(ego_len**2 + ego_wid**2)

        ts_val = float(sdc_row.get("timestamp_seconds", np.nan))

        tgt_df = tgt_grouped.get(t_idx)
        if tgt_df is None or tgt_df.empty:
            continue

        # Extract target arrays
        n = len(tgt_df)
        tcx = tgt_df["center_x"].values.astype(np.float64)
        tcy = tgt_df["center_y"].values.astype(np.float64)
        tvx = tgt_df["velocity_x"].values.astype(np.float64)
        tvy = tgt_df["velocity_y"].values.astype(np.float64)
        tlen = tgt_df["length"].values.astype(np.float64)
        twid = tgt_df["width"].values.astype(np.float64)
        t_ids = tgt_df["track_id"].values
        t_types = tgt_df["object_type"].values

        # Replace NaN velocities with 0
        tvx = np.where(np.isfinite(tvx), tvx, 0.0)
        tvy = np.where(np.isfinite(tvy), tvy, 0.0)

        # Distance filter (vectorized)
        dx = tcx - ego_cx
        dy = tcy - ego_cy
        dist = np.sqrt(dx*dx + dy*dy)
        mask = (dist <= PAIR_RADIUS_M) & np.isfinite(tcx) & np.isfinite(tcy)

        if not mask.any():
            continue

        # Apply mask
        idx = np.where(mask)[0]
        dx = dx[idx]
        dy = dy[idx]
        dist_f = dist[idx]
        dvx = tvx[idx] - ego_vx
        dvy = tvy[idx] - ego_vy
        tlen_f = tlen[idx]
        twid_f = twid[idx]
        t_ids_f = t_ids[idx]
        t_types_f = t_types[idx]

        # Target radii
        tgt_R = 0.5 * np.sqrt(tlen_f**2 + twid_f**2)
        R = ego_R + tgt_R  # combined radii

        # Closing speed (vectorized)
        range_now = dist_f.copy()
        range_now[range_now < 1e-9] = 1e-9
        ux = dx / range_now
        uy = dy / range_now
        closing_speed = -(dvx * ux + dvy * uy)

        # Current overlap
        overlap_now = dist_f <= R

        # Quadratic solve: a*t^2 + b*t + c = 0
        a = dvx*dvx + dvy*dvy
        b = 2.0 * (dx*dvx + dy*dvy)
        c = dx*dx + dy*dy - R*R

        discriminant = b*b - 4.0*a*c

        # Initialize results
        n_pairs = len(idx)
        ttc = np.full(n_pairs, T_MAX_S)
        hit_future = np.zeros(n_pairs, dtype=bool)
        dtc = np.maximum(0.0, dist_f - R)

        # Case 1: already overlapping
        m_overlap = overlap_now
        ttc[m_overlap] = 0.0
        hit_future[m_overlap] = True
        dtc[m_overlap] = 0.0

        # Case 2: has solution (disc >= 0, a > eps)
        m_solve = (~m_overlap) & (discriminant >= 0) & (a > 1e-12)
        if m_solve.any():
            sqrt_d = np.sqrt(discriminant[m_solve])
            a_s = a[m_solve]
            b_s = b[m_solve]

            t1 = (-b_s - sqrt_d) / (2.0 * a_s)
            t2 = (-b_s + sqrt_d) / (2.0 * a_s)

            # First positive root within horizon
            t_hit = np.where(t1 > 0, t1, t2)
            valid_hit = (t_hit > 0) & (t_hit <= T_MAX_S)

            # Build index back into full arrays
            solve_idx = np.where(m_solve)[0]
            ttc[solve_idx[valid_hit]] = t_hit[valid_hit]
            hit_future[solve_idx[valid_hit]] = True
            dtc[solve_idx[valid_hit]] = 0.0

        # status
        status = np.where(
            ~np.isfinite(ttc), "invalid",
            np.where(hit_future | overlap_now, "event", "right_censored")
        )

        # Build rows
        for i in range(n_pairs):
            all_parts.append({
                "scenario_id": scenario_id,
                "time_index": t_idx,
                "timestamp_seconds": ts_val,
                "ego_track_id": sdc_track_id,
                "target_track_id": int(t_ids_f[i]),
                "target_object_type": str(t_types_f[i]),
                "pair_distance_m": float(dist_f[i]),
                "derived_ttc_2d_s": float(ttc[i]),
                "derived_dtc_m": float(dtc[i]),
                "derived_closing_speed_mps": float(closing_speed[i]),
                "derived_overlap_now": bool(overlap_now[i]),
                "derived_hit_future": bool(hit_future[i]),
                "derived_pair_valid": True,
                "derived_ttc_status": str(status[i]),
            })

    if not all_parts:
        return _empty_pair_df()

    return pd.DataFrame(all_parts)


def _empty_pair_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "scenario_id", "time_index", "timestamp_seconds",
        "ego_track_id", "target_track_id", "target_object_type",
        "pair_distance_m",
        "derived_ttc_2d_s", "derived_dtc_m", "derived_closing_speed_mps",
        "derived_overlap_now", "derived_hit_future",
        "derived_pair_valid", "derived_ttc_status",
    ])
