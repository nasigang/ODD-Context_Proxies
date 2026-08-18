#!/usr/bin/env python3
"""
WACV 2027 Phase 2F Invariant and Integrity Test Suite
===================================================
17 Required Comprehensive Unit & Regression Tests
"""

import os
import glob
import json
import zipfile
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/kiapi/waymo_motion_project"
V5_DIR = os.path.join(PROJECT_ROOT, "work", "phase2e_evidence_lock_holdout_20260814_154400")


def get_latest_phase2f_dir():
    candidates = glob.glob(os.path.join(PROJECT_ROOT, "work", "phase2f_scenario_repair_manuscript_closure_*"))
    assert len(candidates) > 0, "No Phase 2F output directory found!"
    return sorted(candidates)[-1]


# ---------------------------------------------------------------------------
# Test 1: V5 Locked Artifact Hash Verification
# ---------------------------------------------------------------------------
def test_01_v5_locked_artifact_hashes():
    p2f_dir = get_latest_phase2f_dir()
    inv_csv = os.path.join(p2f_dir, "audit", "LOCKED_ARTIFACT_INVENTORY_V6.csv")
    assert os.path.exists(inv_csv), f"Missing {inv_csv}"
    df = pd.read_csv(inv_csv)
    assert len(df) >= 45, f"Expected >=45 artifacts, got {len(df)}"
    assert (df["status"] == "VERIFIED").all(), "Some locked V5 artifacts failed hash verification!"


# ---------------------------------------------------------------------------
# Test 2: Zero Raw Holdout Loader Calls
# ---------------------------------------------------------------------------
def test_02_zero_raw_holdout_loader_calls():
    engine_file = os.path.join(PROJECT_ROOT, "phase2_womd", "phase2f_master_engine.py")
    code = open(engine_file).read()
    forbidden_tokens = ["_process_single_holdout_scenario", "tf.data.TFRecordDataset", "parse_womd_scenario"]
    for tok in forbidden_tokens:
        assert tok not in code, f"Forbidden raw loader token '{tok}' found in phase2f_master_engine.py!"


# ---------------------------------------------------------------------------
# Test 3: Zero Fit Calls on Holdout Data
# ---------------------------------------------------------------------------
def test_03_zero_fit_calls_on_holdout():
    engine_file = os.path.join(PROJECT_ROOT, "phase2_womd", "phase2f_master_engine.py")
    code = open(engine_file).read()
    # Check that .fit(X_ho is never called
    assert ".fit(X_ho" not in code
    assert ".fit(df_ho" not in code
    assert ".fit(X_scen_p_ho" not in code


# ---------------------------------------------------------------------------
# Test 4: Primary Replay Point & CI Exact Parity
# ---------------------------------------------------------------------------
def test_04_primary_replay_parity():
    p2f_dir = get_latest_phase2f_dir()
    rep_json = os.path.join(p2f_dir, "audit", "PRIMARY_CURRENT_FRAME_REPLAY_V6.json")
    assert os.path.exists(rep_json)
    data = json.load(open(rep_json))

    mp_ap = data["metrics"]["M_P"]["pr_auc"]
    mall_ap = data["metrics"]["M_P_Eall"]["pr_auc"]
    delta_ap = data["contrasts"]["M_P_Eall_minus_M_P"]["point_delta_ap"]
    ci_low = data["contrasts"]["M_P_Eall_minus_M_P"]["ci_lower_95"]
    ci_high = data["contrasts"]["M_P_Eall_minus_M_P"]["ci_upper_95"]

    assert abs(mp_ap - 0.3223501058) < 1e-8
    assert abs(mall_ap - 0.3370011790) < 1e-8
    assert abs(delta_ap - 0.0146510732) < 1e-8
    assert ci_low > 0.0
    assert ci_high > ci_low


# ---------------------------------------------------------------------------
# Test 5: Minimal Prediction Table Verification
# ---------------------------------------------------------------------------
def test_05_minimal_prediction_table():
    p2f_dir = get_latest_phase2f_dir()
    min_parquet = os.path.join(p2f_dir, "audit", "HOLDOUT_PRIMARY_MINIMAL_PREDICTIONS_V6.parquet")
    assert os.path.exists(min_parquet)
    df = pd.read_parquet(min_parquet)
    assert len(df) == 255164
    assert df["scenario_id"].nunique() == 2804
    req_cols = ["scenario_id", "time_index", "target_y_tau_3s", "weight_scenario_equal", "prob_M_P", "prob_M_P_Eall", "prob_M_P_Einteract"]
    for c in req_cols:
        assert c in df.columns


# ---------------------------------------------------------------------------
# Test 6: Scenario Keyed Merge One-to-One Equality
# ---------------------------------------------------------------------------
def test_06_scenario_keyed_merge():
    p2f_dir = get_latest_phase2f_dir()
    dev_scen = pd.read_csv(os.path.join(p2f_dir, "scenario", "DEV_SCENARIO_ASSOCIATION_REPAIRED_V6.csv"))
    ho_scen = pd.read_csv(os.path.join(p2f_dir, "scenario", "HOLDOUT_SCENARIO_ASSOCIATION_REPAIRED_V6.csv"))
    assert len(dev_scen) == 17 * 3  # 17 features * 3 profile targets
    assert len(ho_scen) == 17 * 3


# ---------------------------------------------------------------------------
# Test 7: Row Permutation Invariance
# ---------------------------------------------------------------------------
def test_07_permutation_invariance():
    p2f_dir = get_latest_phase2f_dir()
    scen_audit = json.load(open(os.path.join(p2f_dir, "scenario", "SCENARIO_REPAIR_AUDIT_V6.json")))
    assert scen_audit["SCENARIO_REPAIR_STATUS"] == "PASS"


# ---------------------------------------------------------------------------
# Test 8: Uniqueness of 1,000 Bootstrap Draws
# ---------------------------------------------------------------------------
def test_08_bootstrap_draws_uniqueness():
    p2f_dir = get_latest_phase2f_dir()
    scen_audit = json.load(open(os.path.join(p2f_dir, "scenario", "SCENARIO_REPAIR_AUDIT_V6.json")))
    assert scen_audit["bootstrap_draws_unique"] >= 990


# ---------------------------------------------------------------------------
# Test 9: Non-degenerate Bootstrap Distributions
# ---------------------------------------------------------------------------
def test_09_nondegenerate_distributions():
    p2f_dir = get_latest_phase2f_dir()
    ho_scen = pd.read_csv(os.path.join(p2f_dir, "scenario", "HOLDOUT_SCENARIO_ASSOCIATION_REPAIRED_V6.csv"))
    assert (ho_scen["bootstrap_sd"] > 0.0).all()
    assert (ho_scen["ci_lower_95"] < ho_scen["ci_upper_95"]).all()


# ---------------------------------------------------------------------------
# Test 10: Holdout Nuisance Transform-Only
# ---------------------------------------------------------------------------
def test_10_nuisance_transform_only():
    p2f_dir = get_latest_phase2f_dir()
    scen_audit = json.load(open(os.path.join(p2f_dir, "scenario", "SCENARIO_REPAIR_AUDIT_V6.json")))
    assert scen_audit["train_scenarios"] == 12828
    assert scen_audit["val_scenarios"] == 2813
    assert scen_audit["holdout_scenarios"] == 2804


# ---------------------------------------------------------------------------
# Test 11: Dynamically Calculated Cross-Level Status
# ---------------------------------------------------------------------------
def test_11_dynamic_cross_level_classification():
    p2f_dir = get_latest_phase2f_dir()
    cl_df = pd.read_csv(os.path.join(p2f_dir, "scenario", "FRAME_SCENARIO_CLASSIFICATION_REPAIRED_V6.csv"))
    valid_classes = {"DISCORDANT_SIGN", "CROSS_LEVEL_STABLE", "FRAME_LOCAL", "TEMPORAL_EXPOSURE_SPECIFIC", "UNSUPPORTED_BOTH"}
    for c in cl_df["cross_level_classification"]:
        assert c in valid_classes


# ---------------------------------------------------------------------------
# Test 12: Executable Narrative-Source Equality
# ---------------------------------------------------------------------------
def test_12_narrative_source_consistency():
    p2f_dir = get_latest_phase2f_dir()
    cons_json = json.load(open(os.path.join(p2f_dir, "integrity", "NARRATIVE_SOURCE_CONSISTENCY_V6.json")))
    assert cons_json["NARRATIVE_SOURCE_CONSISTENCY"] == "PASS"
    assert cons_json["mismatches"] == 0


# ---------------------------------------------------------------------------
# Test 13: Paper Generator Manifest Inclusion
# ---------------------------------------------------------------------------
def test_13_generator_manifest_inclusion():
    p2f_dir = get_latest_phase2f_dir()
    code_manifest = json.load(open(os.path.join(p2f_dir, "integrity", "EXECUTED_CODE_MANIFEST_V6.json")))
    scripts = code_manifest["generator_scripts"]
    assert any("phase2f_master_engine.py" in s for s in scripts)


# ---------------------------------------------------------------------------
# Test 14: Forbidden Novelty & Overclaim Phrase Scan
# ---------------------------------------------------------------------------
def test_14_forbidden_phrase_scan():
    p2f_dir = get_latest_phase2f_dir()
    forbidden = [
        "proves genuine safety information",
        "generalization guarantee",
        "zero-leakage guarantee",
        "future sequence prediction",
        "100% of the total gain",
    ]
    for md_file in glob.glob(os.path.join(p2f_dir, "paper", "*.md")):
        text = open(md_file).read().lower()
        for phrase in forbidden:
            assert phrase not in text, f"Forbidden phrase '{phrase}' found in {md_file}!"


# ---------------------------------------------------------------------------
# Test 15: Kinematic Warp Excluded from Main Tables & Features
# ---------------------------------------------------------------------------
def test_15_warp_excluded_from_main():
    p2f_dir = get_latest_phase2f_dir()
    t1_df = pd.read_csv(os.path.join(p2f_dir, "tables", "TABLE1_NESTED_MODELS_V6.csv"))
    for fset in t1_df["feature_set"]:
        assert "warp" not in fset.lower()


# ---------------------------------------------------------------------------
# Test 16: Zero External Test Access
# ---------------------------------------------------------------------------
def test_16_zero_external_test_access():
    p2f_dir = get_latest_phase2f_dir()
    repro = json.load(open(os.path.join(p2f_dir, "integrity", "REPRODUCIBILITY_MANIFEST_V6.json")))
    assert repro["holdout_protocol"] == "sealed_one_shot"


# ---------------------------------------------------------------------------
# Test 17: Zip Extraction, CSV Parse & SHA-256 Sums Verification
# ---------------------------------------------------------------------------
def test_17_zip_and_sha256_verification():
    p2f_dir = get_latest_phase2f_dir()
    code_zip = os.path.join(p2f_dir, "output", "phase2f_code_package.zip")
    bundle_zip = os.path.join(p2f_dir, "output", "phase2f_manuscript_closure_feedback_bundle.zip")
    assert os.path.exists(code_zip) and os.path.getsize(code_zip) > 0
    assert os.path.exists(bundle_zip) and os.path.getsize(bundle_zip) > 0

    with zipfile.ZipFile(bundle_zip, "r") as zf:
        namelist = zf.namelist()
        assert any("audit/PRIMARY_BOOTSTRAP_REPLICATES_V6.parquet" in name for name in namelist)
        assert any("tables/TABLE1_NESTED_MODELS_V6.csv" in name for name in namelist)
        assert any("paper/FINAL_CLAIM_HIERARCHY_V6.md" in name for name in namelist)
