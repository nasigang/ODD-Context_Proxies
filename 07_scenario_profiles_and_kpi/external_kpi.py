#!/usr/bin/env python3
"""
External KPI Computation
==========================
6 external KPIs for construct validity of the D_s^risk diagnostic vector.
min TTC is explicitly EXCLUDED from external KPIs.

KPIs:
  1. Maximum jerk            — discriminant validity
  2. DRAC                    — convergent / criterion-related
  3. RSS margin              — convergent / criterion-related
  4. Cross-track error       — discriminant validity
  5. TTLC proxy              — discriminant validity
  6. Offroad ratio           — discriminant validity

DRAC and RSS share range/closing_speed with TTC pipeline.
They are NOT labelled "TTC-independent" — classified as convergent.

Usage:
    python phase2_womd/external_kpi.py --max-scenarios 20
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# KPI Registry: formula, source fields, construct category
# ---------------------------------------------------------------------------

KPI_REGISTRY = {
    "max_jerk": {
        "formula": "max(abs(derived_jerk_mps3)) per SDC track per scenario",
        "required_fields": ["derived_jerk_mps3"],
        "source_table": "agent_state",
        "target_shared": False,
        "construct_category": "discriminant",
        "description": "Maximum absolute jerk of SDC trajectory",
    },
    "drac_max": {
        "formula": "max(closing_speed^2 / (2 * range)) per scenario across all pairs",
        "required_fields": ["derived_closing_speed_mps", "pair_distance_m"],
        "source_table": "pair_metrics",
        "target_shared": True,
        "construct_category": "convergent",
        "description": "Maximum Deceleration Rate needed to Avoid Crash",
        "note": "Shares closing_speed with TTC pipeline — criterion-related validity",
    },
    "rss_margin_min": {
        "formula": "min(pair_distance - (v_ego * rho + v_ego^2 / (2 * a_max))) per scenario",
        "required_fields": ["pair_distance_m", "ego_speed_mps"],
        "source_table": "pair_metrics + agent_state",
        "target_shared": True,
        "construct_category": "convergent",
        "description": "Minimum RSS safety margin across all pairs",
        "note": "Shares range data with TTC pipeline — criterion-related validity",
        "params": {"response_time_s": 0.5, "a_max_mps2": 4.0},
    },
    "cross_track_error_max": {
        "formula": "max(min_dist(ego_xy, nearest_lane_centerline)) per scenario",
        "required_fields": ["center_x", "center_y", "lane_polylines"],
        "source_table": "agent_state + map_feature",
        "target_shared": False,
        "construct_category": "discriminant",
        "description": "Maximum cross-track deviation from nearest lane",
    },
    "ttlc_min": {
        "formula": "min(lane_halfwidth / abs(v_lateral)) over exposed frames",
        "required_fields": ["velocity_x", "velocity_y", "heading", "lane_width"],
        "source_table": "agent_state + map_feature",
        "target_shared": False,
        "construct_category": "discriminant",
        "description": "Minimum Time-To-Lane-Crossing proxy",
    },
    "offroad_ratio": {
        "formula": "count(cross_track > road_edge_dist) / total_valid_frames",
        "required_fields": ["center_x", "center_y", "road_edge_polylines"],
        "source_table": "agent_state + map_feature",
        "target_shared": False,
        "construct_category": "discriminant",
        "description": "Fraction of frames where SDC is outside road edges",
    },
}


# ---------------------------------------------------------------------------
# KPI computation functions
# ---------------------------------------------------------------------------

def compute_max_jerk(df_agent: pd.DataFrame, scenario_id: str) -> Dict:
    """KPI 1: Maximum absolute jerk of SDC."""
    sdc = df_agent[
        (df_agent["scenario_id"] == scenario_id) & (df_agent["is_sdc"] == True)
    ]
    jerk = sdc["derived_jerk_mps3"].dropna()
    if len(jerk) == 0:
        return {"value": np.nan, "valid": False, "n_samples": 0}
    return {
        "value": float(jerk.abs().max()),
        "valid": True,
        "n_samples": int(len(jerk)),
    }


def compute_drac_max(df_pairs: pd.DataFrame, scenario_id: str) -> Dict:
    """KPI 2: Maximum DRAC across all pairs."""
    if df_pairs.empty or "scenario_id" not in df_pairs.columns:
        return {"value": np.nan, "valid": False, "n_samples": 0}
    sp = df_pairs[
        (df_pairs["scenario_id"] == scenario_id)
        & (df_pairs.get("derived_pair_valid", pd.Series(dtype=bool)).fillna(False))
    ]
    if sp.empty:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    cs = sp["derived_closing_speed_mps"].values
    dist = sp["pair_distance_m"].values

    # DRAC = cs² / (2d) — only when closing (cs > 0) and d > 0
    valid_mask = (cs > 0) & (dist > 0.1)
    if not valid_mask.any():
        return {"value": 0.0, "valid": True, "n_samples": int(len(sp))}

    drac = cs[valid_mask] ** 2 / (2.0 * dist[valid_mask])
    return {
        "value": float(np.max(drac)),
        "valid": True,
        "n_samples": int(valid_mask.sum()),
    }


def compute_rss_margin_min(
    df_pairs: pd.DataFrame,
    df_agent: pd.DataFrame,
    scenario_id: str,
    rho: float = 0.5,
    a_max: float = 4.0,
) -> Dict:
    """KPI 3: Minimum RSS safety margin."""
    if df_pairs.empty or "scenario_id" not in df_pairs.columns:
        return {"value": np.nan, "valid": False, "n_samples": 0}
    sp = df_pairs[
        (df_pairs["scenario_id"] == scenario_id)
        & (df_pairs["derived_pair_valid"] == True)
    ]
    if sp.empty:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    # Get ego speed per frame
    sdc = df_agent[
        (df_agent["scenario_id"] == scenario_id) & (df_agent["is_sdc"] == True)
    ]
    speed_map = dict(zip(sdc["time_index"], sdc["derived_speed_mps"]))

    margins = []
    for _, row in sp.iterrows():
        v_ego = speed_map.get(row["time_index"], np.nan)
        if not np.isfinite(v_ego):
            continue
        d_actual = row["pair_distance_m"]
        # RSS safe distance: v*ρ + v²/(2a)
        d_rss = v_ego * rho + v_ego ** 2 / (2.0 * a_max)
        margins.append(d_actual - d_rss)

    if not margins:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    return {
        "value": float(np.min(margins)),
        "valid": True,
        "n_samples": int(len(margins)),
    }


def compute_cross_track_error(
    df_agent: pd.DataFrame,
    df_map: pd.DataFrame,
    scenario_id: str,
) -> Dict:
    """KPI 4: Maximum cross-track error from nearest lane centerline."""
    sdc = df_agent[
        (df_agent["scenario_id"] == scenario_id)
        & (df_agent["is_sdc"] == True)
        & (df_agent["valid"] == True)
    ]
    lanes = df_map[
        (df_map["scenario_id"] == scenario_id)
        & (df_map["feature_type"] == "lane")
    ]

    if sdc.empty or lanes.empty:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    # Build lane centerline points array
    lane_pts = lanes[["x", "y"]].values.astype(np.float64)

    ego_xy = sdc[["center_x", "center_y"]].values.astype(np.float64)
    valid_mask = np.isfinite(ego_xy).all(axis=1)
    ego_xy = ego_xy[valid_mask]

    if len(ego_xy) == 0 or len(lane_pts) == 0:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    # Compute min distance from each ego position to nearest lane point
    max_cte = 0.0
    for ex, ey in ego_xy:
        dists = np.sqrt((lane_pts[:, 0] - ex) ** 2 + (lane_pts[:, 1] - ey) ** 2)
        min_d = float(dists.min())
        if min_d > max_cte:
            max_cte = min_d

    return {
        "value": round(max_cte, 4),
        "valid": True,
        "n_samples": int(len(ego_xy)),
    }


def compute_ttlc_min(
    df_agent: pd.DataFrame,
    df_map: pd.DataFrame,
    scenario_id: str,
    default_lane_width: float = 3.7,
) -> Dict:
    """KPI 5: Minimum TTLC proxy (lane_halfwidth / |v_lateral|)."""
    sdc = df_agent[
        (df_agent["scenario_id"] == scenario_id)
        & (df_agent["is_sdc"] == True)
        & (df_agent["valid"] == True)
    ]
    if sdc.empty:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    lane_halfwidth = default_lane_width / 2.0

    ttlc_vals = []
    for _, row in sdc.iterrows():
        vx = row.get("velocity_x", 0)
        vy = row.get("velocity_y", 0)
        heading = row.get("heading", 0)
        if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(heading)):
            continue
        # Lateral speed = velocity component perpendicular to heading
        v_lat = abs(-vx * math.sin(heading) + vy * math.cos(heading))
        if v_lat > 0.1:
            ttlc_vals.append(lane_halfwidth / v_lat)

    if not ttlc_vals:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    return {
        "value": round(float(np.min(ttlc_vals)), 4),
        "valid": True,
        "n_samples": int(len(ttlc_vals)),
    }


def compute_offroad_ratio(
    df_agent: pd.DataFrame,
    df_map: pd.DataFrame,
    scenario_id: str,
    road_edge_threshold: float = 2.0,
) -> Dict:
    """KPI 6: Fraction of frames where SDC is outside road edges."""
    sdc = df_agent[
        (df_agent["scenario_id"] == scenario_id)
        & (df_agent["is_sdc"] == True)
        & (df_agent["valid"] == True)
    ]
    edges = df_map[
        (df_map["scenario_id"] == scenario_id)
        & (df_map["feature_type"] == "road_edge")
    ]

    if sdc.empty or edges.empty:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    edge_pts = edges[["x", "y"]].values.astype(np.float64)
    ego_xy = sdc[["center_x", "center_y"]].values.astype(np.float64)
    valid_mask = np.isfinite(ego_xy).all(axis=1)
    ego_xy = ego_xy[valid_mask]

    if len(ego_xy) == 0:
        return {"value": np.nan, "valid": False, "n_samples": 0}

    n_offroad = 0
    for ex, ey in ego_xy:
        dists = np.sqrt((edge_pts[:, 0] - ex) ** 2 + (edge_pts[:, 1] - ey) ** 2)
        if dists.min() > road_edge_threshold:
            n_offroad += 1

    ratio = n_offroad / len(ego_xy)
    return {
        "value": round(ratio, 4),
        "valid": True,
        "n_samples": int(len(ego_xy)),
    }


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all_kpis(
    df_agent: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_map: pd.DataFrame,
    scenario_ids: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """Compute all 6 external KPIs for given scenarios.

    Returns:
        (kpi_df, coverage_report)
    """
    rows = []
    for sid in scenario_ids:
        row = {"scenario_id": sid}

        kj = compute_max_jerk(df_agent, sid)
        row["max_jerk"] = kj["value"]
        row["max_jerk_valid"] = kj["valid"]

        dr = compute_drac_max(df_pairs, sid)
        row["drac_max"] = dr["value"]
        row["drac_max_valid"] = dr["valid"]

        rss = compute_rss_margin_min(df_pairs, df_agent, sid)
        row["rss_margin_min"] = rss["value"]
        row["rss_margin_min_valid"] = rss["valid"]

        cte = compute_cross_track_error(df_agent, df_map, sid)
        row["cross_track_error_max"] = cte["value"]
        row["cross_track_error_max_valid"] = cte["valid"]

        ttlc = compute_ttlc_min(df_agent, df_map, sid)
        row["ttlc_min"] = ttlc["value"]
        row["ttlc_min_valid"] = ttlc["valid"]

        ofr = compute_offroad_ratio(df_agent, df_map, sid)
        row["offroad_ratio"] = ofr["value"]
        row["offroad_ratio_valid"] = ofr["valid"]

        rows.append(row)

    df_kpi = pd.DataFrame(rows)

    # Coverage report
    coverage = {}
    kpi_names = ["max_jerk", "drac_max", "rss_margin_min",
                 "cross_track_error_max", "ttlc_min", "offroad_ratio"]
    for kpi in kpi_names:
        valid_col = f"{kpi}_valid"
        if valid_col in df_kpi.columns:
            n_valid = int(df_kpi[valid_col].sum())
            n_total = len(df_kpi)
            n_missing = int(df_kpi[kpi].isna().sum())
            coverage[kpi] = {
                **KPI_REGISTRY.get(kpi, {}),
                "coverage": round(n_valid / n_total, 4) if n_total > 0 else 0,
                "missing_rate": round(n_missing / n_total, 4) if n_total > 0 else 0,
                "n_valid": n_valid,
                "n_total": n_total,
            }

    return df_kpi, coverage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="External KPI Computation")
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    parquet_root = os.path.join(args.output_root, "parquet")
    df_agent = pq.read_table(os.path.join(parquet_root, "agent_state")).to_pandas()
    df_map = pq.read_table(os.path.join(parquet_root, "map_feature")).to_pandas()

    # Load pair metrics if available
    pair_path = os.path.join(args.output_root, "pair_metrics.parquet")
    if os.path.isfile(pair_path):
        df_pairs = pd.read_parquet(pair_path)
    else:
        df_pairs = pd.DataFrame()

    sids = df_agent["scenario_id"].unique()
    if args.max_scenarios:
        sids = sids[:args.max_scenarios]

    print(f"[KPI] Computing for {len(sids)} scenarios...")
    df_kpi, coverage = compute_all_kpis(df_agent, df_pairs, df_map, list(sids))

    cov_path = os.path.join(args.output_root, "external_kpi_coverage.json")
    with open(cov_path, "w") as f:
        json.dump(coverage, f, indent=2, default=str)
    print(f"[OK] {cov_path}")

    kpi_path = os.path.join(args.output_root, "external_kpi.parquet")
    df_kpi.to_parquet(kpi_path, index=False)
    print(f"[OK] {kpi_path}")


if __name__ == "__main__":
    main()
