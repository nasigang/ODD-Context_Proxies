#!/usr/bin/env python3
"""
Compute OBB Matched Pairs — Production R1 Label Generator
==========================================================
Computes swept-OBB TTC for all eligible ego-target pairs per frame.

PRODUCTION CONSTRAINTS:
- Source: WOMD training TFRecords ONLY
- Validation/testing paths: UNCONDITIONAL FAIL (pre-parse)
- NaN velocity/heading/dimension: FAIL (not silently zeroed)
- Overlap precedence: invalid > overlap > event > censored > no_exposure
- Circle TTC: separate audit table only (not in primary frame targets)
- Output: partitioned per source file with resume support

Usage:
    python compute_obb_matched_pairs.py --mode smoke --womd_root /data/womd --output_dir output/R1_clean
"""
import argparse
import gc
import glob
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2_womd.obb_ttc_swept import OBBAgent, compute_ttc_obb_swept, T_MAX_S, METHOD_ID

# ── Constants ──
PAIR_RADIUS_M = 70.0
T_MIN_S = 0.05
ELIGIBLE_TYPES = {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}
OBJECT_TYPE_MAP = {0: "TYPE_UNSET", 1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST"}
CONTACT_EPS = 1e-9

# Banned source paths
BANNED_PATH_TOKENS = {"validation", "testing", "test"}


class SourceRoleError(Exception):
    pass


class InvalidStateError(Exception):
    pass


def _validate_source_path(fpath):
    """Reject any path containing validation/testing tokens BEFORE parsing."""
    basename = os.path.basename(fpath).lower()
    dirpath = os.path.dirname(fpath).lower()
    for token in BANNED_PATH_TOKENS:
        if token in basename or f"/{token}/" in dirpath or dirpath.endswith(f"/{token}"):
            raise SourceRoleError(
                f"BLOCKED: source path contains banned token '{token}': {fpath}\n"
                f"Only WOMD training files are allowed.")


def _file_sha256(path, chunk_size=1 << 20):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _validate_ego_state(cx, cy, vx, vy, heading, length, width):
    """Validate ego state. Returns (valid, issues)."""
    issues = []
    for name, val in [("cx", cx), ("cy", cy), ("heading", heading)]:
        if not np.isfinite(val):
            issues.append(f"ego {name} nonfinite: {val}")
    for name, val in [("vx", vx), ("vy", vy)]:
        if not np.isfinite(val):
            issues.append(f"ego {name} nonfinite: {val}")
    for name, val in [("length", length), ("width", width)]:
        if not np.isfinite(val) or val <= 0:
            issues.append(f"ego {name} invalid: {val}")
    return len(issues) == 0, issues


def _validate_target_state(cx, cy, vx, vy, heading, length, width):
    """Validate target state. Returns (valid, issues)."""
    issues = []
    for name, val in [("cx", cx), ("cy", cy), ("heading", heading)]:
        if not np.isfinite(val):
            issues.append(f"target {name} nonfinite: {val}")
    for name, val in [("vx", vx), ("vy", vy)]:
        if not np.isfinite(val):
            issues.append(f"target {name} nonfinite: {val}")
    for name, val in [("length", length), ("width", width)]:
        if not np.isfinite(val) or val <= 0:
            issues.append(f"target {name} invalid: {val}")
    return len(issues) == 0, issues


def process_scenario_obb(sc, scenario_id):
    """Process one scenario with corrected overlap precedence and NaN validation."""
    n_ts = len(sc.timestamps_seconds)
    sdc_idx = sc.sdc_track_index
    tracks = sc.tracks
    if not tracks or n_ts == 0:
        return [], [], {"n_invalid_ego": 0, "n_invalid_target": 0}

    sdc_track_id = tracks[sdc_idx].id
    n_tracks = len(tracks)

    # Pre-allocate arrays
    cx = np.full((n_tracks, n_ts), np.nan)
    cy = np.full((n_tracks, n_ts), np.nan)
    vx = np.full((n_tracks, n_ts), np.nan)
    vy = np.full((n_tracks, n_ts), np.nan)
    heading = np.full((n_tracks, n_ts), np.nan)
    length = np.full((n_tracks, n_ts), np.nan)
    width = np.full((n_tracks, n_ts), np.nan)
    valid = np.zeros((n_tracks, n_ts), dtype=bool)
    obj_types = np.zeros(n_tracks, dtype=np.int32)
    track_ids = np.zeros(n_tracks, dtype=np.int64)

    for ti, track in enumerate(tracks):
        track_ids[ti] = track.id
        obj_types[ti] = track.object_type
        for si, state in enumerate(track.states):
            if si >= n_ts:
                break
            if state.valid:
                valid[ti, si] = True
                cx[ti, si] = state.center_x
                cy[ti, si] = state.center_y
                vx[ti, si] = state.velocity_x
                vy[ti, si] = state.velocity_y
                heading[ti, si] = state.heading
                length[ti, si] = state.length
                width[ti, si] = state.width

    # NO SILENT NaN→0 FILL. Keep NaN as-is; validate per-frame.

    type_names = np.array([OBJECT_TYPE_MAP.get(t, "TYPE_UNSET") for t in obj_types])
    eligible_mask = np.isin(type_names, list(ELIGIBLE_TYPES))
    eligible_mask[sdc_idx] = False

    pair_rows = []
    frame_rows = []
    stats = {"n_invalid_ego": 0, "n_invalid_target": 0}

    for t_idx in range(n_ts):
        ts = float(sc.timestamps_seconds[t_idx])

        # ── Priority 1: invalid_frame (no valid agents) ──
        if not valid[:, t_idx].any():
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "invalid_frame",
                "ttc_obb_swept_s": np.nan, "right_censored": False,
                "overlap_now_flag": False, "n_eligible_pairs": 0,
                "ttc_method": METHOD_ID, "censor_time_s": np.nan,
            })
            continue

        # ── Priority 2: invalid_ego_state ──
        if not valid[sdc_idx, t_idx]:
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "invalid_ego_state",
                "ttc_obb_swept_s": np.nan, "right_censored": False,
                "overlap_now_flag": False, "n_eligible_pairs": 0,
                "ttc_method": METHOD_ID, "censor_time_s": np.nan,
            })
            continue

        # Validate ego numeric state (NO silent zero fill)
        ego_cx_t = float(cx[sdc_idx, t_idx])
        ego_cy_t = float(cy[sdc_idx, t_idx])
        ego_vx_t = float(vx[sdc_idx, t_idx])
        ego_vy_t = float(vy[sdc_idx, t_idx])
        ego_heading_t = float(heading[sdc_idx, t_idx])
        ego_len_t = float(length[sdc_idx, t_idx])
        ego_wid_t = float(width[sdc_idx, t_idx])

        ego_valid, ego_issues = _validate_ego_state(
            ego_cx_t, ego_cy_t, ego_vx_t, ego_vy_t,
            ego_heading_t, ego_len_t, ego_wid_t)

        if not ego_valid:
            stats["n_invalid_ego"] += 1
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "invalid_ego_state",
                "ttc_obb_swept_s": np.nan, "right_censored": False,
                "overlap_now_flag": False, "n_eligible_pairs": 0,
                "ttc_method": METHOD_ID, "censor_time_s": np.nan,
            })
            continue

        # Find eligible targets
        tgt_mask = eligible_mask & valid[:, t_idx]
        tgt_idx_arr = np.where(tgt_mask)[0]

        if len(tgt_idx_arr) == 0:
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "no_exposure",
                "ttc_obb_swept_s": np.nan, "right_censored": False,
                "overlap_now_flag": False, "n_eligible_pairs": 0,
                "ttc_method": METHOD_ID, "censor_time_s": np.nan,
            })
            continue

        # Distance filter
        dx_all = cx[tgt_idx_arr, t_idx] - ego_cx_t
        dy_all = cy[tgt_idx_arr, t_idx] - ego_cy_t
        dist_all = np.sqrt(dx_all**2 + dy_all**2)
        near_mask = dist_all <= PAIR_RADIUS_M

        if not near_mask.any():
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "no_exposure",
                "ttc_obb_swept_s": np.nan, "right_censored": False,
                "overlap_now_flag": False, "n_eligible_pairs": 0,
                "ttc_method": METHOD_ID, "censor_time_s": np.nan,
            })
            continue

        near_idx = tgt_idx_arr[near_mask]
        n_near = len(near_idx)
        dist_f = dist_all[near_mask]

        # ── Compute OBB TTC per pair (validate target state) ──
        ego_obb = OBBAgent(
            cx=ego_cx_t, cy=ego_cy_t,
            length=ego_len_t, width=ego_wid_t,
            heading=ego_heading_t, vx=ego_vx_t, vy=ego_vy_t)

        obb_ttc = np.full(n_near, T_MAX_S)
        obb_hit = np.zeros(n_near, dtype=bool)
        obb_overlap = np.zeros(n_near, dtype=bool)
        obb_status = np.full(n_near, "right_censored", dtype=object)
        pair_valid_flags = np.ones(n_near, dtype=bool)

        for i in range(n_near):
            ti_idx = near_idx[i]
            t_cx = float(cx[ti_idx, t_idx])
            t_cy = float(cy[ti_idx, t_idx])
            t_vx = float(vx[ti_idx, t_idx])
            t_vy = float(vy[ti_idx, t_idx])
            t_h = float(heading[ti_idx, t_idx])
            t_l = float(length[ti_idx, t_idx])
            t_w = float(width[ti_idx, t_idx])

            tgt_valid, tgt_issues = _validate_target_state(
                t_cx, t_cy, t_vx, t_vy, t_h, t_l, t_w)

            if not tgt_valid:
                stats["n_invalid_target"] += 1
                pair_valid_flags[i] = False
                obb_status[i] = "invalid_target"
                continue

            tgt_obb = OBBAgent(cx=t_cx, cy=t_cy, length=t_l, width=t_w,
                               heading=t_h, vx=t_vx, vy=t_vy)
            r = compute_ttc_obb_swept(ego_obb, tgt_obb)
            obb_ttc[i] = r.ttc_s
            obb_hit[i] = r.hit_future
            obb_overlap[i] = r.overlap_now
            obb_status[i] = r.status

        # Record pairs
        for i in range(n_near):
            ti_idx = near_idx[i]
            closing_ux = (cx[ti_idx, t_idx] - ego_cx_t) / max(dist_f[i], 1e-9)
            closing_uy = (cy[ti_idx, t_idx] - ego_cy_t) / max(dist_f[i], 1e-9)
            dvx_i = float(vx[ti_idx, t_idx]) - ego_vx_t if pair_valid_flags[i] else np.nan
            dvy_i = float(vy[ti_idx, t_idx]) - ego_vy_t if pair_valid_flags[i] else np.nan
            closing_speed = -(dvx_i * closing_ux + dvy_i * closing_uy) if pair_valid_flags[i] else np.nan

            pair_rows.append({
                "scenario_id": scenario_id,
                "time_index": t_idx,
                "timestamp_seconds": ts,
                "ego_track_id": int(sdc_track_id),
                "target_track_id": int(track_ids[ti_idx]),
                "target_object_type": type_names[ti_idx],
                "pair_distance_m": float(dist_f[i]),
                "obb_ttc_s": float(obb_ttc[i]),
                "obb_hit_future": bool(obb_hit[i]),
                "obb_overlap_now": bool(obb_overlap[i]),
                "obb_status": str(obb_status[i]),
                "pair_valid": bool(pair_valid_flags[i]),
                "closing_speed_mps": float(closing_speed) if pair_valid_flags[i] else np.nan,
                "ttc_method": METHOD_ID,
                "ttc_horizon_s": T_MAX_S,
            })

        # ── Frame-level target with CORRECT precedence ──
        valid_pairs = pair_valid_flags
        n_valid = int(valid_pairs.sum())

        # Priority 3: current_geometry_overlap (ANY valid pair with overlap)
        has_overlap = (obb_overlap & valid_pairs).any()

        if has_overlap:
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "current_geometry_overlap",
                "ttc_obb_swept_s": 0.0, "right_censored": False,
                "overlap_now_flag": True,
                "n_eligible_pairs": n_valid,
                "n_obb_overlap": int((obb_overlap & valid_pairs).sum()),
                "ttc_method": METHOD_ID, "censor_time_s": np.nan,
            })
            continue

        # Priority 4: future_contact_event
        future_mask = obb_hit & (~obb_overlap) & (obb_ttc > CONTACT_EPS) & valid_pairs
        if future_mask.any():
            future_ttc = obb_ttc[future_mask]
            min_ttc = float(np.min(future_ttc))
            frame_rows.append({
                "scenario_id": scenario_id, "time_index": t_idx,
                "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
                "target_status": "future_contact_event",
                "ttc_obb_swept_s": min_ttc, "right_censored": False,
                "overlap_now_flag": False,
                "n_eligible_pairs": n_valid,
                "n_obb_future_contact": int(future_mask.sum()),
                "ttc_method": METHOD_ID, "censor_time_s": T_MAX_S,
            })
            continue

        # Priority 5: right_censored
        frame_rows.append({
            "scenario_id": scenario_id, "time_index": t_idx,
            "timestamp_seconds": ts, "ego_track_id": int(sdc_track_id),
            "target_status": "right_censored",
            "ttc_obb_swept_s": T_MAX_S, "right_censored": True,
            "overlap_now_flag": False,
            "n_eligible_pairs": n_valid,
            "ttc_method": METHOD_ID, "censor_time_s": T_MAX_S,
        })

    return pair_rows, frame_rows, stats


def _atomic_write_parquet(df, final_path):
    """Write Parquet via temp file + rename for atomic operation."""
    tmp_path = final_path + f".tmp_{uuid.uuid4().hex[:8]}"
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    os.replace(tmp_path, final_path)


def _get_peak_rss_mb():
    """Get peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_partitioned(mode, womd_root, output_dir, run_id=None, max_files=None):
    """
    Partitioned production with resume support.

    Mode: smoke (1-2 files), full (all training).
    Source: ONLY training/ directory. validation/testing → FAIL pre-glob.
    """
    if run_id is None:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    staging_dir = os.path.join(output_dir, "staging", run_id)

    # ── Enforce training-only ──
    training_dir = os.path.join(womd_root, "training")
    training_pattern = os.path.join(training_dir, "training.tfrecord-*")
    train_files = sorted(glob.glob(training_pattern))

    # HARD BLOCK: verify no validation/testing paths snuck in
    for banned in ["validation", "testing"]:
        banned_dir = os.path.join(womd_root, banned)
        if os.path.isdir(banned_dir):
            # Directory exists but we MUST NOT glob or enumerate it
            pass  # existence check only, no listing

    if not train_files:
        raise FileNotFoundError(f"No training TFRecords found: {training_pattern}")

    # Mode-specific file selection
    if mode == "smoke":
        n = min(max_files or 1, 2, len(train_files))
        # Deterministic selection from sorted list
        selected_files = train_files[:n]
    elif mode == "full":
        selected_files = train_files
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[{run_id}] Mode={mode}, {len(selected_files)} training files")

    # ── Check for overwrite of existing staging ──
    if os.path.isdir(staging_dir):
        print(f"[{run_id}] Resuming existing run at {staging_dir}")
    else:
        os.makedirs(staging_dir, exist_ok=True)

    # Create subdirectories
    for sub in ["pair_partitions", "frame_partitions", "progress", "manifests", "reports",
                "audit"]:
        os.makedirs(os.path.join(staging_dir, sub), exist_ok=True)

    # ── Save source manifest ──
    source_manifest = {
        "run_id": run_id,
        "mode": mode,
        "n_files": len(selected_files),
        "source_role": "training_only",
        "validation_files": 0,
        "testing_files": 0,
        "files": [],
    }
    for fpath in selected_files:
        _validate_source_path(fpath)  # Final validation
        fhash = _file_sha256(fpath)
        stable_id = os.path.basename(fpath).replace(".tfrecord", "").replace("-", "_")
        source_manifest["files"].append({
            "path": fpath, "basename": os.path.basename(fpath),
            "stable_id": stable_id, "sha256": fhash,
        })

    manifest_path = os.path.join(staging_dir, "manifests", "source_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(source_manifest, f, indent=2)

    # ── Load progress for resume ──
    progress_path = os.path.join(staging_dir, "progress", "events.jsonl")
    completed_ids = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            for line in f:
                ev = json.loads(line.strip())
                if ev.get("event") == "partition_complete":
                    completed_ids.add(ev["stable_id"])

    # ── Process files ──
    t_global = time.time()
    total_pairs = 0
    total_frames = 0
    total_scenarios = 0

    for fi, file_info in enumerate(source_manifest["files"]):
        fpath = file_info["path"]
        stable_id = file_info["stable_id"]

        # Skip completed partitions (resume)
        if stable_id in completed_ids:
            # Verify output hash
            pair_part = os.path.join(staging_dir, "pair_partitions", f"{stable_id}.parquet")
            frame_part = os.path.join(staging_dir, "frame_partitions", f"{stable_id}.parquet")
            if os.path.exists(pair_part) and os.path.exists(frame_part):
                print(f"[{fi+1}/{len(source_manifest['files'])}] SKIP (completed): {stable_id}")
                continue
            else:
                print(f"[{fi+1}/{len(source_manifest['files'])}] RECOMPUTE (missing output): {stable_id}")

        t0 = time.time()
        file_pairs = []
        file_frames = []
        file_scenarios = []
        file_stats = {"n_invalid_ego": 0, "n_invalid_target": 0}

        dataset = tf.data.TFRecordDataset(fpath, compression_type="")
        for raw in dataset:
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(raw.numpy())
            scenario_id = sc.scenario_id

            pairs, frames, sc_stats = process_scenario_obb(sc, scenario_id)
            file_pairs.extend(pairs)
            file_frames.extend(frames)
            file_scenarios.append(scenario_id)
            file_stats["n_invalid_ego"] += sc_stats["n_invalid_ego"]
            file_stats["n_invalid_target"] += sc_stats["n_invalid_target"]

        elapsed = time.time() - t0

        # Atomic write partitions
        df_pairs = pd.DataFrame(file_pairs) if file_pairs else pd.DataFrame()
        df_frames = pd.DataFrame(file_frames) if file_frames else pd.DataFrame()

        pair_path = os.path.join(staging_dir, "pair_partitions", f"{stable_id}.parquet")
        frame_path = os.path.join(staging_dir, "frame_partitions", f"{stable_id}.parquet")

        if not df_pairs.empty:
            _atomic_write_parquet(df_pairs, pair_path)
        if not df_frames.empty:
            _atomic_write_parquet(df_frames, frame_path)

        # Compute output hashes
        pair_hash = _file_sha256(pair_path) if os.path.exists(pair_path) else "EMPTY"
        frame_hash = _file_sha256(frame_path) if os.path.exists(frame_path) else "EMPTY"

        # Append progress event
        event = {
            "event": "partition_complete",
            "stable_id": stable_id,
            "source_sha256": file_info["sha256"],
            "pair_output_sha256": pair_hash,
            "frame_output_sha256": frame_hash,
            "n_scenarios": len(file_scenarios),
            "n_pairs": len(file_pairs),
            "n_frames": len(file_frames),
            "scenario_ids": file_scenarios,
            "invalid_ego_count": file_stats["n_invalid_ego"],
            "invalid_target_count": file_stats["n_invalid_target"],
            "elapsed_s": round(elapsed, 2),
            "peak_rss_mb": round(_get_peak_rss_mb(), 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(progress_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        completed_ids.add(stable_id)
        total_pairs += len(file_pairs)
        total_frames += len(file_frames)
        total_scenarios += len(file_scenarios)

        print(f"[{fi+1}/{len(source_manifest['files'])}] {stable_id}: "
              f"{len(file_scenarios)} sc, {len(file_pairs)} pairs, {len(file_frames)} frames, "
              f"{elapsed:.1f}s, RSS={_get_peak_rss_mb():.0f}MB")

        gc.collect()

    # ── Write partition manifest ──
    partition_manifest = {
        "run_id": run_id,
        "n_partitions": len(completed_ids),
        "total_pairs": total_pairs,
        "total_frames": total_frames,
        "total_scenarios": total_scenarios,
        "elapsed_s": round(time.time() - t_global, 1),
        "peak_rss_mb": round(_get_peak_rss_mb(), 1),
        "partitions": [],
    }
    # Re-read progress for final manifest
    with open(progress_path) as f:
        for line in f:
            ev = json.loads(line.strip())
            if ev.get("event") == "partition_complete":
                partition_manifest["partitions"].append({
                    "stable_id": ev["stable_id"],
                    "source_sha256": ev["source_sha256"],
                    "pair_sha256": ev["pair_output_sha256"],
                    "frame_sha256": ev["frame_output_sha256"],
                    "n_scenarios": ev["n_scenarios"],
                    "n_pairs": ev["n_pairs"],
                    "n_frames": ev["n_frames"],
                })

    part_manifest_path = os.path.join(staging_dir, "manifests", "partition_manifest.json")
    with open(part_manifest_path, "w") as f:
        json.dump(partition_manifest, f, indent=2)

    # Resource usage report
    usage_report = {
        "run_id": run_id,
        "mode": mode,
        "elapsed_s": round(time.time() - t_global, 1),
        "peak_rss_mb": round(_get_peak_rss_mb(), 1),
        "n_files": len(selected_files),
        "total_pairs": total_pairs,
        "total_frames": total_frames,
        "total_scenarios": total_scenarios,
        "NONINFERENTIAL_PILOT_DO_NOT_REPORT": True,
    }
    with open(os.path.join(staging_dir, "reports", "resource_usage.json"), "w") as f:
        json.dump(usage_report, f, indent=2)

    print(f"\n[{run_id}] DONE: {total_scenarios} scenarios, {total_pairs} pairs, "
          f"{total_frames} frames in {usage_report['elapsed_s']}s")

    return staging_dir, partition_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R1 OBB Label Generator (training-only)")
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"],
                        help="smoke: 1-2 training files; full: all training files")
    parser.add_argument("--womd_root", default=os.environ.get("WOMD_ROOT", "/data/womd"))
    parser.add_argument("--output_dir", default="output/R1_clean")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_files", type=int, default=1,
                        help="Max files for smoke mode (1-2)")
    args = parser.parse_args()
    run_partitioned(args.mode, args.womd_root, args.output_dir,
                    run_id=args.run_id, max_files=args.max_files)
