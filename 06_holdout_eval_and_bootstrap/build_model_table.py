#!/usr/bin/env python3
"""
Model Table Builder
====================
Joins frame_targets + frame_context + scenario-level map/signal aggregates
to produce a model-ready table with C_ODD + Z_state features.

Performs:
  1. Scenario-level split assignment (train/internal_val/internal_holdout/external_test)
  2. Feature engineering from raw tables
  3. Scaler fitting on train split only
  4. Leakage audit against banned column list
  5. Outputs: model_table, train_fitted_stats.json, leakage_audit.json

Usage (inside container):
    python phase2_womd/build_model_table.py --max-scenarios 20
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phase2_womd.obb_ttc import T_MAX_S, T_MIN_S

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_SPEED_LIMIT_MPS = 13.4   # ~30 mph proxy
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "internal_val": 0.15, "internal_holdout": 0.15}

# ---------------------------------------------------------------------------
# Banned columns — target leakage check
# ---------------------------------------------------------------------------
BANNED_COLUMNS: Set[str] = {
    "derived_ttc_2d_s", "ttc_min_s", "y_log_ttc", "ttc_censored",
    "derived_dtc_m", "derived_closing_speed_mps",
    "derived_overlap_now", "derived_hit_future",
    "derived_pair_valid", "derived_ttc_status",
    "n_eligible_pairs", "target_object_id", "target_object_type",
    "target_status", "exposure_flag",
    "derived_accel_mps2", "derived_jerk_mps3",
    "closing_speed", "relative_speed",
    "collision_flag", "overlap_flag", "hit_future",
    "DRAC", "RSS", "jerk", "acceleration",
    "cross_track_error", "best_ttci",
}

BANNED_PATTERNS = ["ttc", "ttci", "collision", "drac", "rss", "closing", "overlap"]

# C_ODD feature names
C_ODD_FEATURES = [
    "odd_n_map_features", "odd_n_lanes", "odd_n_road_lines",
    "odd_n_road_edges", "odd_n_crosswalks", "odd_n_stop_signs",
    "odd_n_speed_bumps", "odd_n_driveways",
    "odd_n_valid_agents", "odd_n_vehicles", "odd_n_pedestrians",
    "odd_n_cyclists",
    "odd_has_signal", "odd_n_active_signals",
    "odd_signal_go_ratio", "odd_signal_stop_ratio",
]

Z_STATE_FEATURES = ["ego_speed_mps", "ego_speed_ratio"]

MODEL_FEATURES = C_ODD_FEATURES + Z_STATE_FEATURES

TARGET_COLUMNS = ["y_log_ttc", "ttc_censored", "exposure_flag", "target_status"]

KEY_COLUMNS = ["scenario_id", "time_index", "timestamp_seconds", "split"]


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_scenario_splits(
    scenario_ids: List[str],
    source_split: str = "training",
) -> Dict[str, str]:
    """Assign scenario-level splits using deterministic hash.

    WOMD training → train/internal_val/internal_holdout
    WOMD validation → external_test
    """
    split_map = {}

    for sid in scenario_ids:
        if source_split == "validation":
            split_map[sid] = "external_test"
        else:
            # Deterministic hash-based split
            h = int(hashlib.md5(
                f"{sid}_{SPLIT_SEED}".encode()
            ).hexdigest(), 16) % 10000
            cum = 0
            frac_train = int(SPLIT_FRACTIONS["train"] * 10000)
            frac_val = int(SPLIT_FRACTIONS["internal_val"] * 10000)

            if h < frac_train:
                split_map[sid] = "train"
            elif h < frac_train + frac_val:
                split_map[sid] = "internal_val"
            else:
                split_map[sid] = "internal_holdout"

    return split_map


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_scenario_map_features(df_map: pd.DataFrame) -> pd.DataFrame:
    """Aggregate map features per scenario into ODD features."""
    if df_map.empty:
        return pd.DataFrame(columns=["scenario_id"] + [
            f for f in C_ODD_FEATURES if f.startswith("odd_n_")
            and f not in ("odd_n_valid_agents", "odd_n_vehicles",
                          "odd_n_pedestrians", "odd_n_cyclists",
                          "odd_n_active_signals")
        ])

    # Count unique features per type per scenario
    feat_counts = df_map.groupby(["scenario_id", "feature_type"])[
        "feature_id"
    ].nunique().unstack(fill_value=0).reset_index()

    result = pd.DataFrame({"scenario_id": feat_counts["scenario_id"]})
    result["odd_n_map_features"] = feat_counts.drop(
        columns=["scenario_id"]
    ).sum(axis=1)

    type_map = {
        "odd_n_lanes": "lane",
        "odd_n_road_lines": "road_line",
        "odd_n_road_edges": "road_edge",
        "odd_n_crosswalks": "crosswalk",
        "odd_n_stop_signs": "stop_sign",
        "odd_n_speed_bumps": "speed_bump",
        "odd_n_driveways": "driveway",
    }

    for col, ft in type_map.items():
        result[col] = feat_counts[ft].values if ft in feat_counts.columns else 0

    return result


def build_scenario_signal_features(
    df_signal: pd.DataFrame,
    n_timestamps_map: Dict[str, int],
) -> pd.DataFrame:
    """Aggregate signal features per scenario+timestep into ODD features."""
    if df_signal.empty:
        return pd.DataFrame(columns=[
            "scenario_id", "time_index",
            "odd_has_signal", "odd_n_active_signals",
            "odd_signal_go_ratio", "odd_signal_stop_ratio",
        ])

    # Per scenario + time_index
    grouped = df_signal.groupby(["scenario_id", "time_index"])

    rows = []
    for (sid, tidx), grp in grouped:
        n_signals = len(grp)
        n_go = grp["signal_state"].isin([
            "LANE_STATE_GO", "LANE_STATE_ARROW_GO"
        ]).sum()
        n_stop = grp["signal_state"].isin([
            "LANE_STATE_STOP", "LANE_STATE_ARROW_STOP",
            "LANE_STATE_FLASHING_STOP",
        ]).sum()

        rows.append({
            "scenario_id": sid,
            "time_index": tidx,
            "odd_has_signal": 1,
            "odd_n_active_signals": n_signals,
            "odd_signal_go_ratio": n_go / n_signals if n_signals > 0 else 0.0,
            "odd_signal_stop_ratio": n_stop / n_signals if n_signals > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def build_ego_state_features(df_agent: pd.DataFrame) -> pd.DataFrame:
    """Extract SDC speed features per frame."""
    sdc = df_agent[df_agent["is_sdc"] == True].copy()
    if sdc.empty:
        return pd.DataFrame(columns=[
            "scenario_id", "time_index",
            "ego_speed_mps", "ego_speed_ratio",
        ])

    result = sdc[["scenario_id", "time_index"]].copy()
    result["ego_speed_mps"] = sdc["derived_speed_mps"].values
    result["ego_speed_ratio"] = (
        sdc["derived_speed_mps"].values / DEFAULT_SPEED_LIMIT_MPS
    )
    return result


# ---------------------------------------------------------------------------
# Scaler fitting (train split only)
# ---------------------------------------------------------------------------

def fit_scalers(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Fit mean/std scalers from train split data only."""
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna()
            stats[col] = {
                "mean": float(vals.mean()) if len(vals) > 0 else 0.0,
                "std": float(vals.std()) if len(vals) > 1 else 1.0,
                "min": float(vals.min()) if len(vals) > 0 else 0.0,
                "max": float(vals.max()) if len(vals) > 0 else 0.0,
                "n_non_null": int(len(vals)),
                "n_total": int(len(df)),
            }
            # Prevent zero std
            if stats[col]["std"] < 1e-12:
                stats[col]["std"] = 1.0
    return stats


def apply_scalers(
    df: pd.DataFrame,
    stats: Dict,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Apply train-fitted z-score scaling."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and col in stats:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            df[f"{col}_z"] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def run_leakage_audit(
    feature_columns: List[str],
    df_columns: List[str],
) -> Dict:
    """Check for target leakage in model input columns.

    Returns audit dict with PASS/FAIL status.
    """
    audit = {
        "status": "PASS",
        "feature_columns_checked": feature_columns,
        "banned_columns_found": [],
        "banned_pattern_matches": [],
        "all_df_columns": df_columns,
    }

    for col in feature_columns:
        canonical = col.lower().strip()

        # Direct match
        if canonical in {b.lower() for b in BANNED_COLUMNS}:
            audit["banned_columns_found"].append(col)

        # Pattern match
        for pat in BANNED_PATTERNS:
            if pat in canonical:
                audit["banned_pattern_matches"].append(
                    {"column": col, "pattern": pat}
                )

    if audit["banned_columns_found"] or audit["banned_pattern_matches"]:
        audit["status"] = "FAIL"

    return audit


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_model_table(
    womd_root: str,
    output_root: str,
    splits: List[str],
    max_scenarios: Optional[int] = None,
) -> Dict:
    """Build model-ready table from parsed Parquet data.

    Steps:
      1. Parse scenarios if parquet not available, or read existing
      2. Build frame targets + pair metrics
      3. Engineer features
      4. Assign splits
      5. Fit scalers on train
      6. Run leakage audit
      7. Write outputs
    """
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    from phase2_womd.parser import parse_scenarios
    from phase2_womd.build_pair_metrics import build_all_pair_metrics
    from phase2_womd.build_frame_targets import build_all_frame_targets

    os.makedirs(output_root, exist_ok=True)
    parquet_root = os.path.join(output_root, "parquet")

    print("=" * 70)
    print("Phase 2 Model Table Builder")
    print("=" * 70)

    # --- Step 1: Parse or load data ---
    scenario_table_path = os.path.join(parquet_root, "scenario_table")
    if os.path.isdir(scenario_table_path):
        print("[Load] Reading existing Parquet tables...")
        df_scenario = pq.read_table(scenario_table_path).to_pandas()
        df_agent = pq.read_table(os.path.join(parquet_root, "agent_state")).to_pandas()
        df_frame = pq.read_table(os.path.join(parquet_root, "frame_context")).to_pandas()
        df_map = pq.read_table(os.path.join(parquet_root, "map_feature")).to_pandas()
        df_signal = pq.read_table(os.path.join(parquet_root, "dynamic_signal")).to_pandas()

        if max_scenarios and len(df_scenario) > max_scenarios:
            keep_ids = df_scenario["scenario_id"].unique()[:max_scenarios]
            df_scenario = df_scenario[df_scenario["scenario_id"].isin(keep_ids)]
            df_agent = df_agent[df_agent["scenario_id"].isin(keep_ids)]
            df_frame = df_frame[df_frame["scenario_id"].isin(keep_ids)]
            df_map = df_map[df_map["scenario_id"].isin(keep_ids)]
            df_signal = df_signal[df_signal["scenario_id"].isin(keep_ids)]
    else:
        print("[Parse] Parsing WOMD TFRecords...")
        parse_scenarios(
            womd_root=womd_root,
            output_root=parquet_root,
            splits=splits,
            max_scenarios=max_scenarios,
        )
        df_scenario = pq.read_table(scenario_table_path).to_pandas()
        df_agent = pq.read_table(os.path.join(parquet_root, "agent_state")).to_pandas()
        df_frame = pq.read_table(os.path.join(parquet_root, "frame_context")).to_pandas()
        df_map = pq.read_table(os.path.join(parquet_root, "map_feature")).to_pandas()
        df_signal = pq.read_table(os.path.join(parquet_root, "dynamic_signal")).to_pandas()

    print(f"  Scenarios: {len(df_scenario)}")
    print(f"  Agent rows: {len(df_agent)}")
    print(f"  Frame rows: {len(df_frame)}")

    # --- Step 2: Build pair metrics + frame targets ---
    print("\n[Step 2] Building pair metrics...")
    df_pairs = build_all_pair_metrics(df_agent, df_scenario)
    print(f"  Pairs: {len(df_pairs)}")

    print("[Step 2b] Building frame targets...")
    df_targets = build_all_frame_targets(df_agent, df_pairs, df_scenario)
    print(f"  Frame targets: {len(df_targets)}")

    # --- Step 3: Feature engineering ---
    print("\n[Step 3] Engineering features...")

    # Map features (scenario-level)
    df_map_agg = build_scenario_map_features(df_map)
    print(f"  Map features aggregated for {len(df_map_agg)} scenarios")

    # Signal features (per frame)
    n_ts_map = dict(zip(df_scenario["scenario_id"], df_scenario["n_timestamps"]))
    df_signal_agg = build_scenario_signal_features(df_signal, n_ts_map)
    print(f"  Signal features: {len(df_signal_agg)} frame-signal rows")

    # Ego state features
    df_ego = build_ego_state_features(df_agent)
    print(f"  Ego state features: {len(df_ego)} rows")

    # --- Step 3b: Merge into model table ---
    print("\n[Step 3b] Merging into model table...")

    # Start with frame targets
    df_model = df_targets.copy()

    # Merge frame_context (traffic counts become ODD features)
    fc_cols = ["scenario_id", "time_index", "n_valid_agents",
               "n_vehicles", "n_pedestrians", "n_cyclists"]
    df_fc_slim = df_frame[fc_cols].copy()
    df_fc_slim = df_fc_slim.rename(columns={
        "n_valid_agents": "odd_n_valid_agents",
        "n_vehicles": "odd_n_vehicles",
        "n_pedestrians": "odd_n_pedestrians",
        "n_cyclists": "odd_n_cyclists",
    })
    df_model = df_model.merge(df_fc_slim, on=["scenario_id", "time_index"], how="left")

    # Merge map features (scenario-level → broadcast to all frames)
    if not df_map_agg.empty:
        df_model = df_model.merge(df_map_agg, on="scenario_id", how="left")
    else:
        for c in ["odd_n_map_features", "odd_n_lanes", "odd_n_road_lines",
                   "odd_n_road_edges", "odd_n_crosswalks", "odd_n_stop_signs",
                   "odd_n_speed_bumps", "odd_n_driveways"]:
            df_model[c] = 0

    # Merge signal features (per frame)
    if not df_signal_agg.empty:
        df_model = df_model.merge(
            df_signal_agg, on=["scenario_id", "time_index"], how="left"
        )
    for c in ["odd_has_signal", "odd_n_active_signals",
              "odd_signal_go_ratio", "odd_signal_stop_ratio"]:
        if c not in df_model.columns:
            df_model[c] = 0
        df_model[c] = df_model[c].fillna(0)

    # Merge ego features
    if not df_ego.empty:
        df_model = df_model.merge(
            df_ego, on=["scenario_id", "time_index"], how="left"
        )
    for c in Z_STATE_FEATURES:
        if c not in df_model.columns:
            df_model[c] = np.nan

    print(f"  Model table: {len(df_model)} rows × {len(df_model.columns)} cols")

    # --- Step 4: Assign splits ---
    print("\n[Step 4] Assigning scenario-level splits...")
    all_sids = df_model["scenario_id"].unique()
    split_map = assign_scenario_splits(list(all_sids), source_split="training")
    df_model["split"] = df_model["scenario_id"].map(split_map)

    split_counts = df_model.groupby("split")["scenario_id"].nunique()
    for sp, cnt in split_counts.items():
        print(f"  {sp}: {cnt} scenarios")

    # --- Step 5: Fit scalers on train only ---
    print("\n[Step 5] Fitting scalers on train split...")
    train_mask = df_model["split"] == "train"
    scaler_stats = fit_scalers(df_model[train_mask], MODEL_FEATURES)

    stats_path = os.path.join(output_root, "train_fitted_stats.json")
    with open(stats_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Apply scalers to all splits
    df_model = apply_scalers(df_model, scaler_stats, MODEL_FEATURES)

    # --- Step 6: Leakage audit ---
    print("\n[Step 6] Running leakage audit...")
    audit = run_leakage_audit(MODEL_FEATURES, list(df_model.columns))

    audit_path = os.path.join(output_root, "leakage_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Audit status: {audit['status']}")
    print(f"  Saved: {audit_path}")

    if audit["status"] == "FAIL":
        print(f"\n[FATAL] LEAKAGE DETECTED!")
        print(f"  Banned columns: {audit['banned_columns_found']}")
        print(f"  Pattern matches: {audit['banned_pattern_matches']}")
        sys.exit(1)

    # --- Step 7: Write model table ---
    print("\n[Step 7] Writing model table...")
    out_cols = KEY_COLUMNS + MODEL_FEATURES + [
        f"{c}_z" for c in MODEL_FEATURES if f"{c}_z" in df_model.columns
    ] + TARGET_COLUMNS

    # Ensure all columns exist
    existing_out = [c for c in out_cols if c in df_model.columns]
    df_out = df_model[existing_out]

    out_path = os.path.join(output_root, "model_table.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df_out)} rows)")

    # --- Summary ---
    summary = {
        "total_frames": len(df_out),
        "total_scenarios": int(df_out["scenario_id"].nunique()),
        "split_scenario_counts": {k: int(v) for k, v in split_counts.items()},
        "feature_columns": MODEL_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "leakage_status": audit["status"],
        "scaler_stats_path": stats_path,
        "model_table_path": out_path,
        "status_distribution": df_out["target_status"].value_counts().to_dict()
        if "target_status" in df_out.columns else {},
    }

    summary_path = os.path.join(output_root, "model_table_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _print_summary(summary)
    return summary


def _print_summary(summary: Dict):
    print("\n" + "=" * 70)
    print("MODEL TABLE SUMMARY")
    print("=" * 70)
    print(f"  Total frames:    {summary['total_frames']}")
    print(f"  Total scenarios: {summary['total_scenarios']}")
    print(f"  Leakage audit:   {summary['leakage_status']}")
    print(f"\n  Split counts:")
    for sp, cnt in summary.get("split_scenario_counts", {}).items():
        print(f"    {sp:20s} {cnt:>6}")
    print(f"\n  Target status distribution:")
    for st, cnt in summary.get("status_distribution", {}).items():
        print(f"    {st:20s} {cnt:>8}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2 Model Table Builder")
    ap.add_argument("--womd-root",
                    default=os.environ.get("WOMD_ROOT", "/mnt/womd"))
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--splits", default="training")
    ap.add_argument("--max-scenarios", type=int, default=None)

    args = ap.parse_args()
    splits = [s.strip() for s in args.splits.split(",")]

    build_model_table(
        womd_root=args.womd_root,
        output_root=args.output_root,
        splits=splits,
        max_scenarios=args.max_scenarios,
    )


if __name__ == "__main__":
    main()
