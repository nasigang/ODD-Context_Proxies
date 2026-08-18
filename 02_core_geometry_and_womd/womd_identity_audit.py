#!/usr/bin/env python3
"""
WOMD v1.3.1 Identity & Schema Audit
====================================
Validates the Waymo Open Motion Dataset file format:
1. File manifest (paths, sizes, SHA-256)
2. Parser identity (Scenario proto vs tf.train.Example)
3. Schema audit (field coverage, timestamps, dt distribution)
4. Feature coverage matrix
5. Unsupported features report

Generates 5 reports under $PHASE2_OUTPUT_ROOT/reports/.

Usage (inside container):
    python phase2_womd/womd_identity_audit.py
"""

import csv
import hashlib
import json
import os
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment paths — container only
# ---------------------------------------------------------------------------
WOMD_ROOT = os.environ.get("WOMD_ROOT", "/mnt/womd")
OUTPUT_ROOT = os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs")
REPORTS_DIR = os.path.join(OUTPUT_ROOT, "reports")

SPLITS = ["training", "validation", "testing"]

# How many scenarios to deep-inspect per split
MAX_DEEP_INSPECT = 50


# ===========================================================================
# Step 1: File Manifest
# ===========================================================================

def build_file_manifest():
    """Walk WOMD_ROOT, collect every file with path/ext/size/sha256."""
    manifest = []
    for split in SPLITS:
        split_dir = os.path.join(WOMD_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory missing: {split_dir}")
            continue
        for fname in sorted(os.listdir(split_dir)):
            fpath = os.path.join(split_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1]
            size = os.path.getsize(fpath)
            sha = _sha256(fpath)
            manifest.append({
                "split": split,
                "filename": fname,
                "extension": ext,
                "size_bytes": size,
                "sha256": sha,
            })
    return manifest


def _sha256(path, chunk_size=1 << 20):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_file_manifest(manifest):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "womd_file_manifest.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "filename", "extension",
                                          "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(manifest)
    print(f"[OK] File manifest written: {out}  ({len(manifest)} files)")
    return out


# ===========================================================================
# Step 2: Parser Identity Testing
# ===========================================================================

def test_parsers():
    """Test Scenario proto and tf.train.Example on first record of each split.

    Returns dict with per-split results and overall adopted parser.
    """
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    results = {}
    overall_scenario_ok = True
    overall_example_ok = True

    for split in SPLITS:
        split_dir = os.path.join(WOMD_ROOT, split)
        if not os.path.isdir(split_dir):
            results[split] = {"error": "directory_missing"}
            overall_scenario_ok = False
            overall_example_ok = False
            continue

        # Find first TFRecord file
        files = sorted([
            f for f in os.listdir(split_dir)
            if "tfrecord" in f.lower()
        ])
        if not files:
            results[split] = {"error": "no_tfrecord_files"}
            overall_scenario_ok = False
            overall_example_ok = False
            continue

        first_file = os.path.join(split_dir, files[0])
        raw_record = None
        for raw in tf.data.TFRecordDataset(first_file):
            raw_record = raw.numpy()
            break

        if raw_record is None:
            results[split] = {"error": "empty_first_file"}
            overall_scenario_ok = False
            overall_example_ok = False
            continue

        # --- Test 1: scenario_pb2.Scenario ---
        scenario_ok = False
        scenario_info = {}
        try:
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(raw_record)
            # Validate key fields are populated
            if sc.scenario_id:
                scenario_ok = True
                scenario_info = {
                    "scenario_id": sc.scenario_id,
                    "n_timestamps": len(sc.timestamps_seconds),
                    "n_tracks": len(sc.tracks),
                    "has_map_features": len(sc.map_features) > 0,
                    "sdc_track_index": sc.sdc_track_index,
                }
        except Exception as e:
            scenario_info = {"error": str(e)}

        # --- Test 2: tf.train.Example ---
        example_ok = False
        example_info = {}
        try:
            ex = tf.train.Example()
            ex.ParseFromString(raw_record)
            feature_keys = list(ex.features.feature.keys())
            if feature_keys:
                example_ok = True
                example_info = {
                    "n_feature_keys": len(feature_keys),
                    "sample_keys": feature_keys[:20],
                }
            else:
                example_info = {"error": "no_features_parsed"}
        except Exception as e:
            example_info = {"error": str(e)}

        if not scenario_ok:
            overall_scenario_ok = False
        if not example_ok:
            overall_example_ok = False

        results[split] = {
            "first_file": files[0],
            "record_size_bytes": len(raw_record),
            "scenario_proto": {"success": scenario_ok, **scenario_info},
            "tf_train_example": {"success": example_ok, **example_info},
        }

    # Decide adopted parser
    if overall_scenario_ok:
        adopted = "scenario_pb2.Scenario"
    elif overall_example_ok:
        adopted = "tf.train.Example"
    else:
        adopted = "NONE"

    return {
        "per_split": results,
        "adopted_parser": adopted,
        "scenario_all_pass": overall_scenario_ok,
        "example_all_pass": overall_example_ok,
    }


# ===========================================================================
# Step 3: Schema Audit (Scenario proto parser)
# ===========================================================================

CORE_FIELDS = [
    "scenario_id",
    "timestamps_seconds",
    "tracks",
    "sdc_track_index",
    "dynamic_map_states",
    "map_features",
    "objects_of_interest",
]

# Fields the user asked us to check but may not exist
OPTIONAL_FIELDS_TO_PROBE = [
    "sdc_paths",
    "path_samples",
]

# Env/weather fields NOT expected in WOMD motion proto
WEATHER_FIELDS = [
    "weather",
    "visibility",
    "friction",
]


def schema_audit_scenario(parser_results):
    """Deep audit using scenario_pb2.Scenario parser."""
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    audit = {
        "parser_type": "scenario_pb2.Scenario",
        "splits": {},
        "timestamp_length_distribution": {},
        "dt_distribution": {},
        "field_coverage": {},
        "unsupported_fields": [],
        "parse_failures": 0,
    }

    all_ts_lengths = []
    all_dts = []
    field_counters = defaultdict(lambda: Counter())  # field -> {present, absent}
    total_scenarios = 0
    per_split_scenario_count = {}

    for split in SPLITS:
        split_dir = os.path.join(WOMD_ROOT, split)
        if not os.path.isdir(split_dir):
            audit["splits"][split] = {"error": "directory_missing"}
            continue

        files = sorted([
            f for f in os.listdir(split_dir)
            if "tfrecord" in f.lower()
        ])

        split_scenarios = 0
        split_failures = 0
        split_ts_lengths = []
        split_dts = []
        split_truncated_files = []
        inspected = 0

        for fname in files:
            fpath = os.path.join(split_dir, fname)
            try:
                for raw in tf.data.TFRecordDataset(fpath):
                    try:
                        sc = scenario_pb2.Scenario()
                        sc.ParseFromString(raw.numpy())
                        split_scenarios += 1
                        total_scenarios += 1

                        if inspected < MAX_DEEP_INSPECT:
                            _inspect_scenario(sc, split, field_counters,
                                              split_ts_lengths, split_dts)
                            inspected += 1
                    except Exception:
                        split_failures += 1
                        audit["parse_failures"] += 1
                        traceback.print_exc()
            except tf.errors.DataLossError as e:
                # Truncated / corrupted TFRecord file — log and continue
                print(f"[WARN] Truncated TFRecord in {split}: {fname} — {e}")
                split_truncated_files.append(fname)
            except Exception as e:
                print(f"[WARN] Error reading {split}/{fname}: {e}")
                split_truncated_files.append(fname)

        all_ts_lengths.extend(split_ts_lengths)
        all_dts.extend(split_dts)
        per_split_scenario_count[split] = split_scenarios

        audit["splits"][split] = {
            "file_count": len(files),
            "scenario_count": split_scenarios,
            "parse_failures": split_failures,
            "deep_inspected": inspected,
            "truncated_files": split_truncated_files,
            "timestamp_lengths": _length_stats(split_ts_lengths),
            "dt_stats": _dt_stats(split_dts),
        }

    # Global distributions
    audit["timestamp_length_distribution"] = _length_stats(all_ts_lengths)
    audit["dt_distribution"] = _dt_stats(all_dts)
    audit["scenario_count_per_split"] = per_split_scenario_count
    audit["total_scenarios"] = total_scenarios

    # Build field coverage
    coverage = {}
    for field in CORE_FIELDS + OPTIONAL_FIELDS_TO_PROBE + WEATHER_FIELDS:
        present = field_counters[field].get("present", 0)
        absent = field_counters[field].get("absent", 0)
        total_checked = present + absent
        coverage[field] = {
            "present": present,
            "absent": absent,
            "total_checked": total_checked,
            "coverage_pct": round(100.0 * present / total_checked, 2) if total_checked > 0 else 0.0,
        }
    audit["field_coverage"] = coverage

    # Unsupported fields
    for field in OPTIONAL_FIELDS_TO_PROBE + WEATHER_FIELDS:
        if coverage.get(field, {}).get("present", 0) == 0:
            audit["unsupported_fields"].append(field)

    return audit


def _inspect_scenario(sc, split, field_counters, ts_lengths, dts):
    """Inspect a single scenario for field presence and timestamp stats."""
    # Core fields presence
    _check_field(sc, "scenario_id", field_counters, lambda s: bool(s.scenario_id))
    _check_field(sc, "timestamps_seconds", field_counters,
                 lambda s: len(s.timestamps_seconds) > 0)
    _check_field(sc, "tracks", field_counters, lambda s: len(s.tracks) > 0)
    _check_field(sc, "sdc_track_index", field_counters,
                 lambda s: True)  # always present (default 0 is valid)
    _check_field(sc, "dynamic_map_states", field_counters,
                 lambda s: len(s.dynamic_map_states) > 0)
    _check_field(sc, "map_features", field_counters,
                 lambda s: len(s.map_features) > 0)
    _check_field(sc, "objects_of_interest", field_counters,
                 lambda s: len(s.objects_of_interest) > 0)

    # Optional / non-existent fields
    _check_field(sc, "sdc_paths", field_counters,
                 lambda s: hasattr(s, "sdc_paths") and bool(getattr(s, "sdc_paths", None)))
    _check_field(sc, "path_samples", field_counters,
                 lambda s: hasattr(s, "path_samples") and bool(getattr(s, "path_samples", None)))

    # Weather / env fields (not expected)
    for wf in WEATHER_FIELDS:
        _check_field(sc, wf, field_counters,
                     lambda s, f=wf: hasattr(s, f) and bool(getattr(s, f, None)))

    # Timestamps
    ts = list(sc.timestamps_seconds)
    ts_lengths.append(len(ts))
    if len(ts) >= 2:
        for i in range(1, len(ts)):
            dts.append(round(ts[i] - ts[i - 1], 6))


def _check_field(sc, name, counters, check_fn):
    try:
        if check_fn(sc):
            counters[name]["present"] += 1
        else:
            counters[name]["absent"] += 1
    except Exception:
        counters[name]["absent"] += 1


def _length_stats(lengths):
    if not lengths:
        return {}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "median": round(statistics.median(lengths), 2),
        "unique_values": sorted(set(lengths)),
        "count": len(lengths),
    }


def _dt_stats(dts):
    if not dts:
        return {}
    return {
        "min": round(min(dts), 6),
        "max": round(max(dts), 6),
        "mean": round(statistics.mean(dts), 6),
        "median": round(statistics.median(dts), 6),
        "stdev": round(statistics.stdev(dts), 6) if len(dts) > 1 else 0.0,
        "count": len(dts),
    }


# ===========================================================================
# Step 4: Write Reports
# ===========================================================================

def write_identity_report(parser_results, audit, manifest):
    """womd_identity.json"""
    split_file_counts = {}
    for split in SPLITS:
        split_file_counts[split] = sum(
            1 for m in manifest if m["split"] == split
        )

    version_evidence = []
    for split, info in parser_results["per_split"].items():
        if isinstance(info, dict) and "scenario_proto" in info:
            sp = info["scenario_proto"]
            if sp.get("success"):
                version_evidence.append(
                    f"{split}: Scenario proto parsed OK, "
                    f"scenario_id={sp.get('scenario_id', '?')}, "
                    f"n_timestamps={sp.get('n_timestamps', '?')}"
                )

    report = {
        "status": "PASS" if parser_results["adopted_parser"] != "NONE" and audit["parse_failures"] == 0 else "FAIL",
        "parser_type": parser_results["adopted_parser"],
        "detected_version_evidence": version_evidence,
        "package_version": "waymo-open-dataset-tf-2-12-0==1.6.7",
        "split_file_counts": split_file_counts,
        "scenario_count_per_split": audit.get("scenario_count_per_split", {}),
        "total_scenarios": audit.get("total_scenarios", 0),
        "timestamp_length_distribution": audit.get("timestamp_length_distribution", {}),
        "dt_distribution": audit.get("dt_distribution", {}),
        "field_coverage_summary": {
            k: v["coverage_pct"]
            for k, v in audit.get("field_coverage", {}).items()
        },
        "parse_failures": audit.get("parse_failures", 0),
        "unsupported_features": audit.get("unsupported_fields", []),
        "source_file_hashes": {
            m["filename"]: m["sha256"] for m in manifest[:10]
        },
        "source_file_hash_note": f"Showing first 10 of {len(manifest)} files. Full hashes in womd_file_manifest.csv.",
    }

    out = os.path.join(REPORTS_DIR, "womd_identity.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Identity report: {out}")
    return report


def write_schema_audit(audit):
    """womd_schema_audit.json"""
    out = os.path.join(REPORTS_DIR, "womd_schema_audit.json")
    with open(out, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"[OK] Schema audit: {out}")


def write_feature_coverage(audit):
    """womd_feature_coverage.csv"""
    out = os.path.join(REPORTS_DIR, "womd_feature_coverage.csv")
    fc = audit.get("field_coverage", {})
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "present", "absent", "total_checked", "coverage_pct", "status"])
        for field in CORE_FIELDS + OPTIONAL_FIELDS_TO_PROBE + WEATHER_FIELDS:
            info = fc.get(field, {})
            present = info.get("present", 0)
            absent = info.get("absent", 0)
            total = info.get("total_checked", 0)
            pct = info.get("coverage_pct", 0.0)
            status = "supported" if present > 0 else "unsupported"
            w.writerow([field, present, absent, total, pct, status])
    print(f"[OK] Feature coverage: {out}")


def write_unsupported_features(audit):
    """womd_unsupported_features.md"""
    out = os.path.join(REPORTS_DIR, "womd_unsupported_features.md")
    unsup = audit.get("unsupported_fields", [])
    lines = [
        "# WOMD v1.3.1 Unsupported Features Report",
        "",
        f"Generated by `womd_identity_audit.py`",
        "",
        "## Summary",
        "",
        f"The following {len(unsup)} field(s) were requested but **not found** "
        f"in any inspected scenario record:",
        "",
    ]
    if unsup:
        for field in unsup:
            reason = _unsupported_reason(field)
            lines.append(f"- **`{field}`** — {reason}")
    else:
        lines.append("- *(none — all requested fields were present)*")

    lines.extend([
        "",
        "## Implications",
        "",
        "Fields marked as unsupported should NOT be used in downstream "
        "processing (TTC calculation, model training). Any pipeline code "
        "that references these fields must handle their absence gracefully.",
        "",
        "## Field Details",
        "",
        "| Field | Status | Notes |",
        "|-------|--------|-------|",
    ])
    for field in CORE_FIELDS + OPTIONAL_FIELDS_TO_PROBE + WEATHER_FIELDS:
        fc = audit.get("field_coverage", {}).get(field, {})
        present = fc.get("present", 0)
        status = "✅ Supported" if present > 0 else "❌ Unsupported"
        notes = _unsupported_reason(field) if present == 0 else f"{fc.get('coverage_pct', 0)}% coverage"
        lines.append(f"| `{field}` | {status} | {notes} |")

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Unsupported features: {out}")


def _unsupported_reason(field):
    reasons = {
        "sdc_paths": "Not part of WOMD Motion Scenario proto schema",
        "path_samples": "Not part of WOMD Motion Scenario proto schema",
        "weather": "WOMD Motion does not include weather metadata",
        "visibility": "WOMD Motion does not include visibility metadata",
        "friction": "WOMD Motion does not include friction metadata",
    }
    return reasons.get(field, "Field not found in parsed records")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("WOMD v1.3.1 Identity & Schema Audit")
    print("=" * 70)
    print(f"WOMD_ROOT    = {WOMD_ROOT}")
    print(f"OUTPUT_ROOT  = {OUTPUT_ROOT}")
    print(f"REPORTS_DIR  = {REPORTS_DIR}")
    print()

    # Verify input paths
    if not os.path.isdir(WOMD_ROOT):
        print(f"[FATAL] WOMD_ROOT does not exist: {WOMD_ROOT}")
        sys.exit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: File manifest
    # ------------------------------------------------------------------
    print("[Step 1/5] Building file manifest...")
    manifest = build_file_manifest()
    write_file_manifest(manifest)
    print(f"  Total files: {len(manifest)}")
    for split in SPLITS:
        n = sum(1 for m in manifest if m["split"] == split)
        print(f"  {split}: {n} files")
    print()

    # ------------------------------------------------------------------
    # Step 2: Parser identity testing
    # ------------------------------------------------------------------
    print("[Step 2/5] Testing parsers (Scenario proto & tf.train.Example)...")
    parser_results = test_parsers()
    print(f"  Scenario proto all-pass: {parser_results['scenario_all_pass']}")
    print(f"  tf.train.Example all-pass: {parser_results['example_all_pass']}")
    print(f"  Adopted parser: {parser_results['adopted_parser']}")
    for split, info in parser_results["per_split"].items():
        if isinstance(info, dict):
            sp = info.get("scenario_proto", {})
            ex = info.get("tf_train_example", {})
            print(f"  [{split}] Scenario={sp.get('success', '?')}  "
                  f"Example={ex.get('success', '?')}  "
                  f"file={info.get('first_file', '?')}")
    print()

    if parser_results["adopted_parser"] == "NONE":
        print("[FATAL] No parser succeeded on all splits. Halting.")
        _write_fail_reports(parser_results, manifest)
        sys.exit(1)

    if parser_results["adopted_parser"] != "scenario_pb2.Scenario":
        print(f"[WARN] Non-Scenario parser adopted: {parser_results['adopted_parser']}")
        print("  Schema audit designed for Scenario proto — skipping deep audit.")
        _write_fail_reports(parser_results, manifest)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Schema audit
    # ------------------------------------------------------------------
    print("[Step 3/5] Running schema audit (this reads ALL records)...")
    audit = schema_audit_scenario(parser_results)
    print(f"  Total scenarios parsed: {audit['total_scenarios']}")
    print(f"  Parse failures: {audit['parse_failures']}")
    print(f"  Timestamp length distribution: {audit['timestamp_length_distribution']}")
    print(f"  dt distribution: {audit['dt_distribution']}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Write all reports
    # ------------------------------------------------------------------
    print("[Step 4/5] Writing reports...")
    identity_report = write_identity_report(parser_results, audit, manifest)
    write_schema_audit(audit)
    write_feature_coverage(audit)
    write_unsupported_features(audit)
    print()

    # ------------------------------------------------------------------
    # Step 5: Final verdict
    # ------------------------------------------------------------------
    print("[Step 5/5] Final verdict")
    print("=" * 70)
    status = identity_report["status"]
    parser = identity_report["parser_type"]

    if status == "PASS":
        print(f"RESULT         = PASS")
        print(f"PARSER_TYPE    = {parser}")
        print(f"TOTAL_SCENARIOS= {audit['total_scenarios']}")
        print(f"PARSE_FAILURES = {audit['parse_failures']}")
        print(f"UNSUPPORTED    = {audit.get('unsupported_fields', [])}")
        print(f"NEXT_STEP      = Proceed with {parser} for all downstream "
              f"processing (TTC, model training). "
              f"Do NOT use dataset_pb2.Frame.")
    else:
        print(f"RESULT         = FAIL")
        print(f"PARSER_TYPE    = {parser}")
        print(f"PARSE_FAILURES = {audit['parse_failures']}")
        print(f"NEXT_STEP      = DO NOT proceed with model training or TTC "
              f"calculation. Fix parser/data issues first.")
    print("=" * 70)
    sys.exit(0 if status == "PASS" else 1)


def _write_fail_reports(parser_results, manifest):
    """Write minimal reports when we can't do full audit."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fail_report = {
        "status": "FAIL",
        "parser_type": parser_results["adopted_parser"],
        "detail": parser_results["per_split"],
        "parse_failures": "N/A — parser identity failed",
    }
    with open(os.path.join(REPORTS_DIR, "womd_identity.json"), "w") as f:
        json.dump(fail_report, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "womd_schema_audit.json"), "w") as f:
        json.dump({"status": "FAIL", "reason": "parser_identity_failed"}, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "womd_feature_coverage.csv"), "w") as f:
        f.write("field,present,absent,total_checked,coverage_pct,status\n")
    with open(os.path.join(REPORTS_DIR, "womd_unsupported_features.md"), "w") as f:
        f.write("# WOMD Unsupported Features\n\nAudit FAILED: parser identity could not be established.\n")
    print("[OK] Fail reports written.")


if __name__ == "__main__":
    main()
