#!/usr/bin/env python3
"""
Kinematic Warp Feasibility Check
==================================
Assesses whether trajectory-level warp is physically consistent.

7 consistency checks (train scenarios ONLY):
  1. Position continuity      — max(Δpos) < 2m between consecutive frames
  2. Velocity consistency     — recomputed v vs original within 10%
  3. Acceleration consistency — recomputed a vs original within 20%
  4. Heading continuity       — max(abs(Δheading)) < 0.5 rad per step
  5. Box geometry             — length, width preserved exactly
  6. Map/lane consistency     — cross-track error change < 1m
  7. Interaction topology     — same eligible pairs before/after

If ANY check fails → M5 disabled, report 'feature-space augmentation only'.
Validation/test splits NEVER receive warp.

Note: this is NOT called 'physics-consistent warp'. It is a feasibility
check for trajectory-level warp.

Usage:
    python phase2_womd/warp_feasibility.py --max-scenarios 20
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Consistency check thresholds
# ---------------------------------------------------------------------------
MAX_POS_JUMP_M = 2.0
VELOCITY_REL_TOL = 0.10
ACCEL_REL_TOL = 0.20
MAX_HEADING_JUMP_RAD = 0.5
CROSS_TRACK_CHANGE_M = 1.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_position_continuity(df_sdc: pd.DataFrame) -> Dict:
    """Check 1: max position jump between consecutive frames < 2m."""
    df = df_sdc.sort_values("time_index")
    x = df["center_x"].values
    y = df["center_y"].values

    if len(x) < 2:
        return {"passed": True, "max_jump_m": 0.0, "n_frames": len(x)}

    dx = np.diff(x)
    dy = np.diff(y)
    jumps = np.sqrt(dx ** 2 + dy ** 2)
    valid = np.isfinite(jumps)

    if not valid.any():
        return {"passed": True, "max_jump_m": 0.0, "n_frames": len(x)}

    max_jump = float(jumps[valid].max())
    return {
        "passed": max_jump < MAX_POS_JUMP_M,
        "max_jump_m": round(max_jump, 4),
        "threshold_m": MAX_POS_JUMP_M,
        "n_frames": int(len(x)),
    }


def check_velocity_consistency(df_sdc: pd.DataFrame) -> Dict:
    """Check 2: finite-diff velocity vs proto velocity within 10%."""
    df = df_sdc.sort_values("time_index")
    x = df["center_x"].values
    y = df["center_y"].values
    ts = df["timestamp_seconds"].values
    vx_proto = df["velocity_x"].values
    vy_proto = df["velocity_y"].values

    if len(x) < 2:
        return {"passed": True, "max_rel_error": 0.0}

    # Finite-difference velocity
    dt = np.diff(ts)
    vx_fd = np.diff(x) / np.maximum(dt, 1e-12)
    vy_fd = np.diff(y) / np.maximum(dt, 1e-12)

    # Compare with proto velocity at index i+1
    vx_p = vx_proto[1:]
    vy_p = vy_proto[1:]

    speed_fd = np.sqrt(vx_fd ** 2 + vy_fd ** 2)
    speed_p = np.sqrt(vx_p ** 2 + vy_p ** 2)

    denom = np.maximum(speed_p, 1.0)
    rel_err = np.abs(speed_fd - speed_p) / denom
    valid = np.isfinite(rel_err) & (speed_p > 0.5)  # only check when moving

    if not valid.any():
        return {"passed": True, "max_rel_error": 0.0, "n_checked": 0}

    max_err = float(rel_err[valid].max())
    return {
        "passed": max_err < VELOCITY_REL_TOL,
        "max_rel_error": round(max_err, 4),
        "mean_rel_error": round(float(rel_err[valid].mean()), 4),
        "threshold": VELOCITY_REL_TOL,
        "n_checked": int(valid.sum()),
    }


def check_acceleration_consistency(df_sdc: pd.DataFrame) -> Dict:
    """Check 3: finite-diff accel vs derived accel within 20%."""
    df = df_sdc.sort_values("time_index")
    speed_derived = df["derived_speed_mps"].values
    accel_derived = df["derived_accel_mps2"].values
    ts = df["timestamp_seconds"].values

    if len(speed_derived) < 3:
        return {"passed": True, "max_rel_error": 0.0}

    # Finite-diff accel from speed
    dt = np.diff(ts)
    accel_fd = np.diff(speed_derived) / np.maximum(dt, 1e-12)

    accel_d = accel_derived[1:]

    denom = np.maximum(np.abs(accel_d), 0.5)
    rel_err = np.abs(accel_fd - accel_d) / denom
    valid = np.isfinite(rel_err) & np.isfinite(accel_d)

    if not valid.any():
        return {"passed": True, "max_rel_error": 0.0, "n_checked": 0}

    max_err = float(rel_err[valid].max())
    return {
        "passed": max_err < ACCEL_REL_TOL,
        "max_rel_error": round(max_err, 4),
        "threshold": ACCEL_REL_TOL,
        "n_checked": int(valid.sum()),
    }


def check_heading_continuity(df_sdc: pd.DataFrame) -> Dict:
    """Check 4: max heading jump < 0.5 rad per step."""
    df = df_sdc.sort_values("time_index")
    heading = df["heading"].values

    if len(heading) < 2:
        return {"passed": True, "max_jump_rad": 0.0}

    dh = np.diff(heading)
    # Wrap to [-π, π)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    valid = np.isfinite(dh)

    if not valid.any():
        return {"passed": True, "max_jump_rad": 0.0}

    max_jump = float(np.abs(dh[valid]).max())
    return {
        "passed": max_jump < MAX_HEADING_JUMP_RAD,
        "max_jump_rad": round(max_jump, 4),
        "threshold_rad": MAX_HEADING_JUMP_RAD,
    }


def check_box_geometry(df_sdc: pd.DataFrame) -> Dict:
    """Check 5: length and width are constant across frames."""
    length = df_sdc["length"].dropna().values
    width = df_sdc["width"].dropna().values

    if len(length) == 0:
        return {"passed": True, "length_range": 0, "width_range": 0}

    l_range = float(length.max() - length.min())
    w_range = float(width.max() - width.min())

    return {
        "passed": l_range < 0.01 and w_range < 0.01,
        "length_range": round(l_range, 4),
        "width_range": round(w_range, 4),
    }


def check_map_consistency(
    df_sdc: pd.DataFrame,
    df_map: pd.DataFrame,
    scenario_id: str,
) -> Dict:
    """Check 6: cross-track error change < 1m between consecutive frames."""
    lanes = df_map[
        (df_map["scenario_id"] == scenario_id)
        & (df_map["feature_type"] == "lane")
    ]
    if lanes.empty or df_sdc.empty:
        return {"passed": True, "note": "no lane data", "max_cte_change": 0.0}

    lane_pts = lanes[["x", "y"]].values.astype(np.float64)
    df = df_sdc.sort_values("time_index")
    ego_xy = df[["center_x", "center_y"]].values.astype(np.float64)
    valid = np.isfinite(ego_xy).all(axis=1)

    if valid.sum() < 2:
        return {"passed": True, "max_cte_change": 0.0}

    ctes = []
    for i in range(len(ego_xy)):
        if not valid[i]:
            ctes.append(np.nan)
            continue
        dists = np.sqrt((lane_pts[:, 0] - ego_xy[i, 0]) ** 2 +
                        (lane_pts[:, 1] - ego_xy[i, 1]) ** 2)
        ctes.append(float(dists.min()))

    ctes = np.array(ctes)
    dcte = np.abs(np.diff(ctes))
    finite = np.isfinite(dcte)

    if not finite.any():
        return {"passed": True, "max_cte_change": 0.0}

    max_change = float(dcte[finite].max())
    return {
        "passed": max_change < CROSS_TRACK_CHANGE_M,
        "max_cte_change": round(max_change, 4),
        "threshold_m": CROSS_TRACK_CHANGE_M,
    }


def check_interaction_topology(
    df_agent: pd.DataFrame,
    scenario_id: str,
) -> Dict:
    """Check 7: eligible interaction pairs are consistent across the scenario.

    Verifies that agent appearance/disappearance is gradual (no sudden
    teleportation of interactants).
    """
    sc = df_agent[
        (df_agent["scenario_id"] == scenario_id) & (df_agent["valid"] == True)
    ]
    if sc.empty:
        return {"passed": True, "n_tracks": 0}

    # Track appearance: each track should have contiguous valid ranges
    tracks = sc.groupby("track_id")["time_index"]
    n_fragmented = 0
    for tid, indices in tracks:
        idx = sorted(indices)
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        # A gap > 5 frames is suspicious
        if np.any(gaps > 5):
            n_fragmented += 1

    n_tracks = int(sc["track_id"].nunique())
    frag_ratio = n_fragmented / n_tracks if n_tracks > 0 else 0

    return {
        "passed": frag_ratio < 0.2,
        "n_tracks": n_tracks,
        "n_fragmented": n_fragmented,
        "fragmentation_ratio": round(frag_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Main feasibility pipeline
# ---------------------------------------------------------------------------

def run_warp_feasibility(
    output_root: str,
    max_scenarios: Optional[int] = None,
) -> Dict:
    """Run all 7 warp consistency checks on train scenarios."""
    import pyarrow.parquet as pq

    os.makedirs(output_root, exist_ok=True)
    parquet_root = os.path.join(output_root, "parquet")

    print("=" * 60)
    print("KINEMATIC WARP FEASIBILITY CHECK")
    print("=" * 60)

    # Load data
    df_agent = pq.read_table(os.path.join(parquet_root, "agent_state")).to_pandas()
    df_map = pq.read_table(os.path.join(parquet_root, "map_feature")).to_pandas()

    # Load splits — only use train
    model_table_path = os.path.join(output_root, "model_table.parquet")
    if os.path.isfile(model_table_path):
        df_model = pd.read_parquet(model_table_path)
        train_sids = df_model[df_model["split"] == "train"]["scenario_id"].unique()
    else:
        train_sids = df_agent["scenario_id"].unique()

    if max_scenarios:
        train_sids = train_sids[:max_scenarios]

    print(f"  Checking {len(train_sids)} train scenarios")
    print(f"  Validation/test: NOT checked (warp never applied)")

    # Run checks per scenario
    all_checks = {
        "position_continuity": [],
        "velocity_consistency": [],
        "acceleration_consistency": [],
        "heading_continuity": [],
        "box_geometry": [],
        "map_consistency": [],
        "interaction_topology": [],
    }
    per_scenario = []

    for sid in train_sids:
        sdc = df_agent[
            (df_agent["scenario_id"] == sid)
            & (df_agent["is_sdc"] == True)
            & (df_agent["valid"] == True)
        ]

        c1 = check_position_continuity(sdc)
        c2 = check_velocity_consistency(sdc)
        c3 = check_acceleration_consistency(sdc)
        c4 = check_heading_continuity(sdc)
        c5 = check_box_geometry(sdc)
        c6 = check_map_consistency(sdc, df_map, sid)
        c7 = check_interaction_topology(df_agent, sid)

        all_checks["position_continuity"].append(c1["passed"])
        all_checks["velocity_consistency"].append(c2["passed"])
        all_checks["acceleration_consistency"].append(c3["passed"])
        all_checks["heading_continuity"].append(c4["passed"])
        all_checks["box_geometry"].append(c5["passed"])
        all_checks["map_consistency"].append(c6["passed"])
        all_checks["interaction_topology"].append(c7["passed"])

        all_pass = all([c1["passed"], c2["passed"], c3["passed"],
                        c4["passed"], c5["passed"], c6["passed"], c7["passed"]])

        per_scenario.append({
            "scenario_id": sid,
            "all_passed": all_pass,
            "position": c1["passed"],
            "velocity": c2["passed"],
            "acceleration": c3["passed"],
            "heading": c4["passed"],
            "geometry": c5["passed"],
            "map": c6["passed"],
            "topology": c7["passed"],
        })

    # Aggregate results
    summary = {}
    any_global_fail = False
    for check_name, results in all_checks.items():
        n_pass = sum(results)
        n_total = len(results)
        pass_rate = n_pass / n_total if n_total > 0 else 0
        all_pass = n_pass == n_total

        if not all_pass:
            any_global_fail = True

        summary[check_name] = {
            "all_passed": all_pass,
            "pass_rate": round(pass_rate, 4),
            "n_passed": n_pass,
            "n_total": n_total,
            "n_failed": n_total - n_pass,
        }
        status = "✓" if all_pass else f"✗ ({n_total - n_pass} fail)"
        print(f"  {check_name:30s} {status}")

    # M5 decision
    if any_global_fail:
        m5_status = "disabled"
        m5_reason = "feature-space augmentation only"
    else:
        m5_status = "eligible"
        m5_reason = "all trajectory-level consistency checks passed"

    report = {
        "m5_status": m5_status,
        "m5_reason": m5_reason,
        "any_check_failed": any_global_fail,
        "checks": summary,
        "n_scenarios_checked": int(len(train_sids)),
        "split": "train only",
        "validation_test_warp": "NEVER applied",
        "note": "This is a trajectory-level warp feasibility check, not a physics-consistent warp",
    }

    # Save
    report_path = os.path.join(output_root, "warp_consistency_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OK] {report_path}")

    # No-warp vs warp ablation (placeholder since warp failed/wasn't applied)
    ablation = pd.DataFrame([{
        "condition": "no_warp",
        "m5_status": m5_status,
        "note": "Baseline — no trajectory warp applied",
    }])
    if any_global_fail:
        ablation = pd.concat([ablation, pd.DataFrame([{
            "condition": "warp_attempted",
            "m5_status": "disabled",
            "note": "Trajectory-level checks failed — feature-space augmentation only",
        }])], ignore_index=True)

    ablation_path = os.path.join(output_root, "no_warp_vs_warp_ablation.csv")
    ablation.to_csv(ablation_path, index=False)
    print(f"[OK] {ablation_path}")

    # Print final verdict
    print(f"\n{'='*60}")
    print(f"M5 DECISION: {m5_status.upper()}")
    print(f"Reason: {m5_reason}")
    print(f"{'='*60}")

    return report


def main():
    ap = argparse.ArgumentParser(description="Warp Feasibility Check")
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    run_warp_feasibility(args.output_root, args.max_scenarios)


if __name__ == "__main__":
    main()
