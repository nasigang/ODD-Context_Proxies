#!/usr/bin/env python3
"""
WOMD Scenario→Parquet Parser
==============================
Main orchestrator that reads TFRecord files containing scenario_pb2.Scenario
records and converts them into role-separated, scenario_id-partitioned
Parquet tables.

Usage (inside container):
    python phase2_womd/parser.py \\
        --womd-root /mnt/womd \\
        --output-root /mnt/phase2_outputs/parquet \\
        --splits training,validation \\
        --max-scenarios 10

Tables produced:
    scenario_table.parquet/   (partitioned by scenario_id)
    agent_state.parquet/
    frame_context.parquet/
    map_feature.parquet/
    dynamic_signal.parquet/
    sdc_path_points.parquet/  (only if coverage > 0)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phase2_womd.kinematics import compute_kinematics
from phase2_womd.map_features import extract_map_features, get_map_coverage_stats
from phase2_womd.schema import (
    AGENT_STATE_PRIMARY_KEY,
    AGENT_STATE_SCHEMA,
    DYNAMIC_SIGNAL_PRIMARY_KEY,
    DYNAMIC_SIGNAL_SCHEMA,
    FRAME_CONTEXT_PRIMARY_KEY,
    FRAME_CONTEXT_SCHEMA,
    MAP_FEATURE_PRIMARY_KEY,
    MAP_FEATURE_SCHEMA,
    OBJECT_TYPE_MAP,
    SCENARIO_TABLE_SCHEMA,
    SDC_PATH_PRIMARY_KEY,
    SDC_PATH_SCHEMA,
    UNSUPPORTED_FIELDS,
)
from phase2_womd.signal_features import (
    extract_dynamic_signals,
    get_signal_coverage_stats,
)


# ---------------------------------------------------------------------------
# Scenario extraction
# ---------------------------------------------------------------------------

def extract_scenario_row(scenario, scenario_id: str) -> Dict:
    """Build one row for scenario_table from a Scenario proto."""
    ts = list(scenario.timestamps_seconds)
    n_ts = len(ts)
    has_sdc_paths = (
        hasattr(scenario, "sdc_paths")
        and len(getattr(scenario, "sdc_paths", [])) > 0
    )

    return {
        "scenario_id": scenario_id,
        "segment_id": scenario_id,   # alias
        "n_timestamps": n_ts,
        "n_tracks": len(scenario.tracks),
        "sdc_track_index": scenario.sdc_track_index,
        "n_map_features": len(scenario.map_features),
        "n_dynamic_signal_steps": len(scenario.dynamic_map_states),
        "n_objects_of_interest": len(scenario.objects_of_interest),
        "has_sdc_paths": has_sdc_paths,
        "timestamp_start": ts[0] if ts else None,
        "timestamp_end": ts[-1] if ts else None,
        "duration_seconds": (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0,
    }


def extract_agent_states(scenario, scenario_id: str) -> List[Dict]:
    """Extract per-track, per-timestep raw state rows.

    Rule: invalid states are NOT filled with 0 — raw proto values are kept,
    but downstream operations guard on 'valid'.
    """
    timestamps = list(scenario.timestamps_seconds)
    objects_of_interest = set(scenario.objects_of_interest)
    sdc_idx = scenario.sdc_track_index
    rows = []

    for track_idx, track in enumerate(scenario.tracks):
        track_id = track.id
        obj_type = OBJECT_TYPE_MAP.get(track.object_type, f"UNKNOWN_{track.object_type}")
        is_sdc = (track_idx == sdc_idx)
        is_ooi = (track_id in objects_of_interest)

        for t_idx, state in enumerate(track.states):
            ts_val = timestamps[t_idx] if t_idx < len(timestamps) else None

            if state.valid:
                row = {
                    "scenario_id": scenario_id,
                    "segment_id": scenario_id,
                    "time_index": t_idx,
                    "frame_label": t_idx,
                    "timestamp_seconds": ts_val,
                    "track_id": track_id,
                    "obj_id": track_id,
                    "object_type": obj_type,
                    "valid": True,
                    "center_x": state.center_x,
                    "center_y": state.center_y,
                    "center_z": state.center_z,
                    "velocity_x": state.velocity_x,
                    "velocity_y": state.velocity_y,
                    "heading": state.heading,
                    "length": state.length,
                    "width": state.width,
                    "height": state.height,
                    "is_sdc": is_sdc,
                    "is_object_of_interest": is_ooi,
                }
            else:
                # Invalid state — store NaN for numeric fields, NOT zero
                row = {
                    "scenario_id": scenario_id,
                    "segment_id": scenario_id,
                    "time_index": t_idx,
                    "frame_label": t_idx,
                    "timestamp_seconds": ts_val,
                    "track_id": track_id,
                    "obj_id": track_id,
                    "object_type": obj_type,
                    "valid": False,
                    "center_x": np.nan,
                    "center_y": np.nan,
                    "center_z": np.nan,
                    "velocity_x": np.nan,
                    "velocity_y": np.nan,
                    "heading": np.nan,
                    "length": np.nan,
                    "width": np.nan,
                    "height": np.nan,
                    "is_sdc": is_sdc,
                    "is_object_of_interest": is_ooi,
                }

            rows.append(row)

    return rows


def extract_frame_context(scenario, scenario_id: str) -> List[Dict]:
    """Build frame_context rows — per-timestep agent counts.

    NOTE: No TTC, range, closing_speed, or collision flags here.
    """
    timestamps = list(scenario.timestamps_seconds)
    sdc_idx = scenario.sdc_track_index
    n_timestamps = len(timestamps)
    rows = []

    for t_idx in range(n_timestamps):
        n_valid = 0
        n_veh = 0
        n_ped = 0
        n_cyc = 0
        n_oth = 0

        for track in scenario.tracks:
            if t_idx < len(track.states) and track.states[t_idx].valid:
                n_valid += 1
                ot = track.object_type
                if ot == 1:
                    n_veh += 1
                elif ot == 2:
                    n_ped += 1
                elif ot == 3:
                    n_cyc += 1
                else:
                    n_oth += 1

        rows.append({
            "scenario_id": scenario_id,
            "segment_id": scenario_id,
            "time_index": t_idx,
            "frame_label": t_idx,
            "timestamp_seconds": timestamps[t_idx],
            "n_valid_agents": n_valid,
            "n_vehicles": n_veh,
            "n_pedestrians": n_ped,
            "n_cyclists": n_cyc,
            "n_other": n_oth,
        })

    return rows


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def check_duplicates(df: pd.DataFrame, keys: List[str], table_name: str) -> int:
    """Check for duplicate primary keys. Returns count of duplicates."""
    if df.empty:
        return 0
    dupes = df.duplicated(subset=keys, keep=False).sum()
    if dupes > 0:
        print(f"  [WARN] {table_name}: {dupes} duplicate key rows "
              f"on {keys}")
    return dupes


def check_foreign_keys(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    key: str,
    child_name: str,
    parent_name: str,
) -> int:
    """Check that all child key values exist in parent. Returns orphan count."""
    if child_df.empty or parent_df.empty:
        return 0
    child_vals = set(child_df[key].unique())
    parent_vals = set(parent_df[key].unique())
    orphans = child_vals - parent_vals
    if orphans:
        print(f"  [WARN] {child_name}.{key} has {len(orphans)} values "
              f"not in {parent_name}")
    return len(orphans)


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

def write_partitioned_parquet(
    df: pd.DataFrame,
    schema: pa.Schema,
    output_dir: str,
    table_name: str,
    partition_col: str = "scenario_id",
):
    """Write DataFrame as scenario_id-partitioned Parquet."""
    if df.empty:
        print(f"  [SKIP] {table_name}: empty DataFrame")
        return

    out_path = os.path.join(output_dir, table_name)
    os.makedirs(out_path, exist_ok=True)

    # Align columns to schema, filling missing with None
    schema_names = [f.name for f in schema]
    for col in schema_names:
        if col not in df.columns:
            df[col] = None

    # Select only schema columns in order
    df_ordered = df[schema_names]

    table = pa.Table.from_pandas(df_ordered, schema=schema, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=[partition_col],
    )
    print(f"  [OK] {table_name}: {len(df)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Main parse pipeline
# ---------------------------------------------------------------------------

def parse_scenarios(
    womd_root: str,
    output_root: str,
    splits: List[str],
    max_scenarios: Optional[int] = None,
) -> Dict:
    """Parse WOMD TFRecords into Parquet tables.

    Returns a summary dict with row counts, coverage stats, etc.
    """
    # Lazy import TF / proto to avoid import errors outside container
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    os.makedirs(output_root, exist_ok=True)

    # Accumulators
    all_scenario_rows = []
    all_agent_rows = []
    all_frame_rows = []
    all_map_rows = []
    all_signal_rows = []
    all_sdc_path_rows = []

    total_parsed = 0
    total_failures = 0
    scenario_ids_seen = set()

    print("=" * 70)
    print("WOMD Scenario→Parquet Parser")
    print("=" * 70)
    print(f"  WOMD_ROOT   = {womd_root}")
    print(f"  OUTPUT_ROOT = {output_root}")
    print(f"  SPLITS      = {splits}")
    print(f"  MAX_SCEN    = {max_scenarios or 'ALL'}")
    print()

    for split in splits:
        split_dir = os.path.join(womd_root, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory missing: {split_dir}")
            continue

        files = sorted([
            f for f in os.listdir(split_dir)
            if "tfrecord" in f.lower()
        ])
        print(f"[{split}] {len(files)} TFRecord files")

        for fname in files:
            if max_scenarios and total_parsed >= max_scenarios:
                break

            fpath = os.path.join(split_dir, fname)
            try:
                for raw in tf.data.TFRecordDataset(fpath):
                    if max_scenarios and total_parsed >= max_scenarios:
                        break

                    try:
                        sc = scenario_pb2.Scenario()
                        sc.ParseFromString(raw.numpy())
                        sid = sc.scenario_id

                        if sid in scenario_ids_seen:
                            print(f"  [WARN] Duplicate scenario_id: {sid}")
                            continue
                        scenario_ids_seen.add(sid)

                        # Extract all tables
                        all_scenario_rows.append(
                            extract_scenario_row(sc, sid)
                        )
                        all_agent_rows.extend(
                            extract_agent_states(sc, sid)
                        )
                        all_frame_rows.extend(
                            extract_frame_context(sc, sid)
                        )
                        all_map_rows.extend(
                            extract_map_features(sc, sid)
                        )
                        all_signal_rows.extend(
                            extract_dynamic_signals(sc, sid)
                        )

                        # SDC paths — only if present
                        if (hasattr(sc, "sdc_paths")
                                and len(getattr(sc, "sdc_paths", [])) > 0):
                            for path in sc.sdc_paths:
                                for pt_idx, pt in enumerate(path):
                                    all_sdc_path_rows.append({
                                        "scenario_id": sid,
                                        "point_index": pt_idx,
                                        "x": pt.x,
                                        "y": pt.y,
                                        "z": pt.z,
                                    })

                        total_parsed += 1
                        if total_parsed % 100 == 0:
                            print(f"  ... parsed {total_parsed} scenarios")

                    except Exception as e:
                        total_failures += 1
                        print(f"  [ERR] Parse failure in {fname}: {e}")

            except Exception as e:
                print(f"  [WARN] Error reading {split}/{fname}: {e}")

        if max_scenarios and total_parsed >= max_scenarios:
            print(f"  Reached max_scenarios={max_scenarios}, stopping.")
            break

    print(f"\nTotal parsed: {total_parsed}  |  Failures: {total_failures}")
    print()

    # --- Build DataFrames ---
    print("[Building DataFrames]")
    df_scenario = pd.DataFrame(all_scenario_rows)
    df_agent = pd.DataFrame(all_agent_rows)
    df_frame = pd.DataFrame(all_frame_rows)
    df_map = pd.DataFrame(all_map_rows)
    df_signal = pd.DataFrame(all_signal_rows)

    # --- Compute derived kinematics ---
    if not df_agent.empty:
        print("[Computing derived kinematics]")
        t0 = time.time()
        df_agent = compute_kinematics(df_agent)
        print(f"  Kinematics done in {time.time() - t0:.1f}s")

    # --- Integrity checks ---
    print("\n[Integrity Checks]")
    summary = {
        "scenarios_parsed": total_parsed,
        "parse_failures": total_failures,
        "tables": {},
        "duplicate_keys": {},
        "foreign_key_orphans": 0,
    }

    tables_info = [
        ("scenario_table", df_scenario, AGENT_STATE_PRIMARY_KEY[:1],
         SCENARIO_TABLE_SCHEMA),
        ("agent_state", df_agent, AGENT_STATE_PRIMARY_KEY,
         AGENT_STATE_SCHEMA),
        ("frame_context", df_frame, FRAME_CONTEXT_PRIMARY_KEY,
         FRAME_CONTEXT_SCHEMA),
        ("map_feature", df_map, MAP_FEATURE_PRIMARY_KEY,
         MAP_FEATURE_SCHEMA),
        ("dynamic_signal", df_signal, DYNAMIC_SIGNAL_PRIMARY_KEY,
         DYNAMIC_SIGNAL_SCHEMA),
    ]

    for tname, df, keys, schema in tables_info:
        row_count = len(df)
        dupes = check_duplicates(df, keys, tname)
        summary["tables"][tname] = {"row_count": row_count}
        summary["duplicate_keys"][tname] = dupes
        print(f"  {tname}: {row_count} rows, {dupes} duplicate keys")

    # Foreign key checks
    if not df_agent.empty and not df_scenario.empty:
        orphans = check_foreign_keys(
            df_agent, df_scenario, "scenario_id",
            "agent_state", "scenario_table"
        )
        summary["foreign_key_orphans"] += orphans

    # --- Invalid state ratio ---
    if not df_agent.empty:
        n_total = len(df_agent)
        n_invalid = (~df_agent["valid"]).sum()
        summary["invalid_state_ratio"] = round(n_invalid / n_total, 4)
        print(f"  invalid_state_ratio: {summary['invalid_state_ratio']:.4f} "
              f"({n_invalid}/{n_total})")

        # Verify invalid states are NOT zero
        invalid_rows = df_agent[~df_agent["valid"]]
        if not invalid_rows.empty:
            center_x_vals = invalid_rows["center_x"]
            zeros_in_invalid = (center_x_vals == 0.0).sum()
            nans_in_invalid = center_x_vals.isna().sum()
            print(f"  invalid center_x: {nans_in_invalid} NaN, "
                  f"{zeros_in_invalid} zero (should be all NaN)")
            summary["invalid_states_are_nan"] = bool(
                nans_in_invalid == len(invalid_rows)
            )
    else:
        summary["invalid_state_ratio"] = None

    # --- Coverage stats ---
    summary["map_coverage"] = get_map_coverage_stats(all_map_rows)
    summary["signal_coverage"] = get_signal_coverage_stats(
        all_signal_rows,
        sum(r["n_timestamps"] for r in all_scenario_rows) if all_scenario_rows else 0,
    )
    summary["route_coverage"] = {
        "sdc_path_points": len(all_sdc_path_rows),
        "status": "unsupported" if len(all_sdc_path_rows) == 0 else "present",
    }
    summary["unsupported_features"] = UNSUPPORTED_FIELDS

    # --- Missing feature coverage ---
    if not df_agent.empty:
        feature_cols = [
            "center_x", "center_y", "center_z",
            "velocity_x", "velocity_y", "heading",
            "length", "width", "height",
        ]
        missing_pct = {}
        for col in feature_cols:
            if col in df_agent.columns:
                valid_rows = df_agent[df_agent["valid"]]
                if not valid_rows.empty:
                    miss = valid_rows[col].isna().sum()
                    missing_pct[col] = round(
                        100.0 * miss / len(valid_rows), 4
                    )
                else:
                    missing_pct[col] = None
        summary["missing_feature_coverage"] = missing_pct

    # --- Write Parquet ---
    print(f"\n[Writing Parquet to {output_root}]")
    write_partitioned_parquet(
        df_scenario, SCENARIO_TABLE_SCHEMA, output_root, "scenario_table"
    )
    write_partitioned_parquet(
        df_agent, AGENT_STATE_SCHEMA, output_root, "agent_state"
    )
    write_partitioned_parquet(
        df_frame, FRAME_CONTEXT_SCHEMA, output_root, "frame_context"
    )
    write_partitioned_parquet(
        df_map, MAP_FEATURE_SCHEMA, output_root, "map_feature"
    )
    write_partitioned_parquet(
        df_signal, DYNAMIC_SIGNAL_SCHEMA, output_root, "dynamic_signal"
    )

    # SDC paths only if coverage > 0
    if all_sdc_path_rows:
        df_sdc = pd.DataFrame(all_sdc_path_rows)
        write_partitioned_parquet(
            df_sdc, SDC_PATH_SCHEMA, output_root, "sdc_path_points"
        )
    else:
        print("  [SKIP] sdc_path_points: 0% coverage (unsupported)")

    # --- Write summary report ---
    report_path = os.path.join(output_root, "parser_smoke_test_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Summary report: {report_path}")

    # --- Print summary ---
    _print_summary(summary)

    return summary


def _print_summary(summary: Dict):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("PARSE SUMMARY")
    print("=" * 70)
    print(f"  Scenarios parsed:    {summary['scenarios_parsed']}")
    print(f"  Parse failures:      {summary['parse_failures']}")
    print()
    print("  Table Row Counts:")
    for tname, info in summary.get("tables", {}).items():
        print(f"    {tname:20s}  {info['row_count']:>10,}")
    print()
    print("  Duplicate Keys:")
    for tname, count in summary.get("duplicate_keys", {}).items():
        status = "✓" if count == 0 else f"✗ {count}"
        print(f"    {tname:20s}  {status}")
    print()
    if summary.get("invalid_state_ratio") is not None:
        print(f"  Invalid state ratio: {summary['invalid_state_ratio']:.4f}")
    print()
    print("  Map Coverage:")
    mc = summary.get("map_coverage", {})
    print(f"    Features: {mc.get('n_features', 0)}, "
          f"Points: {mc.get('total_points', 0)}")
    if mc.get("feature_types"):
        for ft, cnt in mc["feature_types"].items():
            print(f"      {ft}: {cnt}")
    print()
    print("  Signal Coverage:")
    sc = summary.get("signal_coverage", {})
    print(f"    Entries: {sc.get('total_entries', 0)}, "
          f"Unique lanes: {sc.get('unique_lanes', 0)}, "
          f"Coverage: {sc.get('signal_coverage_pct', 0)}%")
    print()
    print("  Route Coverage:")
    rc = summary.get("route_coverage", {})
    print(f"    Status: {rc.get('status', '?')}, "
          f"Points: {rc.get('sdc_path_points', 0)}")
    print()
    print("  Unsupported Features:")
    for f in summary.get("unsupported_features", []):
        print(f"    - {f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WOMD Scenario→Parquet Parser"
    )
    parser.add_argument(
        "--womd-root", default=os.environ.get("WOMD_ROOT", "/mnt/womd"),
        help="Path to WOMD data root containing training/validation/testing",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(
            os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
            "parquet",
        ),
        help="Output directory for Parquet tables",
    )
    parser.add_argument(
        "--splits", default="training",
        help="Comma-separated list of splits to process",
    )
    parser.add_argument(
        "--max-scenarios", type=int, default=None,
        help="Maximum number of scenarios to process (for smoke testing)",
    )

    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",")]

    summary = parse_scenarios(
        womd_root=args.womd_root,
        output_root=args.output_root,
        splits=splits,
        max_scenarios=args.max_scenarios,
    )

    # Exit non-zero if any failures or duplicate keys
    has_dupes = any(v > 0 for v in summary.get("duplicate_keys", {}).values())
    if summary["parse_failures"] > 0 or has_dupes:
        print("\n[FAIL] Issues detected — do NOT proceed with full data processing.")
        sys.exit(1)
    else:
        print("\n[PASS] Smoke test succeeded.")
        sys.exit(0)


if __name__ == "__main__":
    main()
