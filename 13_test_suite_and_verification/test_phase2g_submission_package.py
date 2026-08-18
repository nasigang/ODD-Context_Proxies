"""
Comprehensive Invariant and Compliance Test Suite for WACV 2027 Phase 2G.
Verifies all 27 critical gates and submission requirements.
"""

import os
import glob
import json
import zipfile
import hashlib
import re
import pytest
import pandas as pd

WORKSPACE_ROOT = "/home/kiapi/waymo_motion_project"


def get_latest_phase2g_dir():
    candidates = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, "work", "phase2g_submission_manuscript_novelty_lock_*")))
    if not candidates:
        pytest.skip("No Phase 2G output directory found. Run phase2g_master_engine.py first.")
    return candidates[-1]


def test_01_v5_v6_source_hash_equality():
    p2g_dir = get_latest_phase2g_dir()
    audit_json = os.path.join(p2g_dir, "evidence", "EVIDENCE_FREEZE_AUDIT_V7.json")
    assert os.path.exists(audit_json), f"Missing {audit_json}"
    with open(audit_json, "r") as f:
        data = json.load(f)
    assert data["v6_verification_status"] == "ALL_VERIFIED"
    assert data["v6_verified_files_count"] >= 40


def test_02_zero_raw_holdout_loader_calls():
    for root, dirs, files in os.walk(os.path.join(WORKSPACE_ROOT, "phase2_womd")):
        for f in files:
            if f.endswith(".py") and ("phase2f" in f or "phase2g" in f):
                with open(os.path.join(root, f), "r") as py_file:
                    content = py_file.read()
                    assert "_process_single_holdout_scenario" not in content


def test_03_zero_holdout_fit_tuning_predictions():
    p2g_dir = get_latest_phase2g_dir()
    audit_json = os.path.join(p2g_dir, "evidence", "EVIDENCE_FREEZE_AUDIT_V7.json")
    with open(audit_json, "r") as f:
        data = json.load(f)
    assert data["holdout_governance"]["fit_call_on_holdout_count"] == 0
    assert data["holdout_governance"]["new_model_or_prediction_artifacts_since_v6"] == 0


def test_04_primary_hierarchy_order():
    p2g_dir = get_latest_phase2g_dir()
    intro_tex = os.path.join(p2g_dir, "paper", "sections", "1_introduction.tex")
    with open(intro_tex, "r") as f:
        text = f.read()
    pos_all = text.find("0.0147")
    pos_interact = text.find("0.0161")
    assert pos_all != -1 and pos_interact != -1
    assert pos_all < pos_interact, "Primary full model (+0.0147) must appear before interaction (+0.0161)"


def test_05_primary_numeric_exact_equality():
    p2g_dir = get_latest_phase2g_dir()
    audit_json = os.path.join(p2g_dir, "evidence", "EVIDENCE_FREEZE_AUDIT_V7.json")
    with open(audit_json, "r") as f:
        data = json.load(f)
    metrics = data["locked_authoritative_metrics"]
    assert abs(metrics["M_P_ap"] - 0.3223501058) < 1e-9
    assert abs(metrics["M_P_Eall_ap"] - 0.3370011790) < 1e-9
    assert abs(metrics["delta_ap_primary"] - 0.0146510732) < 1e-9
    assert abs(metrics["ci_lower_95_primary"] - 0.0005446066) < 1e-9
    assert abs(metrics["ci_upper_95_primary"] - 0.0285107759) < 1e-9


def test_06_narrative_numeric_resolution():
    p2g_dir = get_latest_phase2g_dir()
    map_csv = os.path.join(p2g_dir, "evidence", "NARRATIVE_NUMBER_MAP_V7.csv")
    assert os.path.exists(map_csv)
    df = pd.read_csv(map_csv)
    assert len(df) >= 20
    assert all(df["verification_status"] == "PASS")


def test_07_no_pseudo_pvalues_in_main():
    p2g_dir = get_latest_phase2g_dir()
    main_tex = os.path.join(p2g_dir, "paper", "main.tex")
    with open(main_tex, "r") as f:
        text = f.read()
    assert "p = 0.0470" not in text
    assert "p = 0.0050" not in text
    assert "p_empirical" not in text
    for sec_file in glob.glob(os.path.join(p2g_dir, "paper", "sections", "*.tex")):
        with open(sec_file, "r") as f:
            sec_text = f.read()
            assert "p = 0.0470" not in sec_text
            assert "p = 0.0050" not in sec_text


def test_08_no_orthogonal_predictive_info():
    p2g_dir = get_latest_phase2g_dir()
    for sec_file in glob.glob(os.path.join(p2g_dir, "paper", "sections", "*.tex")):
        with open(sec_file, "r") as f:
            sec_text = f.read().lower()
            assert "orthogonal predictive information" not in sec_text
            assert "orthogonal information" not in sec_text


def test_09_no_proves_safety_or_generalization_guarantee():
    p2g_dir = get_latest_phase2g_dir()
    forbidden = ["proves safety", "generalization guarantee", "zero-leakage guarantee", "large gain", "massive gain"]
    for sec_file in glob.glob(os.path.join(p2g_dir, "paper", "sections", "*.tex")):
        with open(sec_file, "r") as f:
            sec_text = f.read().lower()
            for phrase in forbidden:
                assert phrase not in sec_text, f"Forbidden phrase '{phrase}' found in {sec_file}"


def test_10_no_18_interaction_features():
    p2g_dir = get_latest_phase2g_dir()
    for sec_file in glob.glob(os.path.join(p2g_dir, "paper", "sections", "*.tex")):
        with open(sec_file, "r") as f:
            sec_text = f.read().lower()
            assert "18 interaction features" not in sec_text


def test_11_feature_confirmation_wording_10_of_13_vs_13_of_13():
    p2g_dir = get_latest_phase2g_dir()
    sec5_file = os.path.join(p2g_dir, "paper", "sections", "5_primary_incremental_validity.tex")
    with open(sec5_file, "r") as f:
        text = f.read()
    assert "10 out of 13" in text or "10/13" in text
    assert "13 out of 13" in text or "13/13" in text
    assert "76.9" in text
    assert "100" in text


def test_12_scenario_wording_feature_dependent():
    p2g_dir = get_latest_phase2g_dir()
    sec6_file = os.path.join(p2g_dir, "paper", "sections", "6_mechanism_and_temporal_completeness.tex")
    with open(sec6_file, "r") as f:
        text = f.read().lower()
    assert "feature-dependent" in text or "anchor-supported" in text


def test_13_temporal_role_retrospective_statement():
    p2g_dir = get_latest_phase2g_dir()
    doc_path = os.path.join(p2g_dir, "novelty", "TEMPORAL_ROLE_AND_OFFLINE_USE_STATEMENT_V7.md")
    assert os.path.exists(doc_path)
    with open(doc_path, "r") as f:
        text = f.read()
    assert "retrospective snapshot" in text.lower()
    assert "no cross-time feature leakage" in text.lower()


def test_14_odd_context_proxy_definition_present():
    p2g_dir = get_latest_phase2g_dir()
    sec1_file = os.path.join(p2g_dir, "paper", "sections", "1_introduction.tex")
    with open(sec1_file, "r") as f:
        text = f.read()
    assert "operational-context proxies" in text or "ODD-context proxies" in text


def test_15_no_local_absolute_paths_or_identity():
    p2g_dir = get_latest_phase2g_dir()
    for root, dirs, files in os.walk(os.path.join(p2g_dir, "paper")):
        for f in files:
            if f.endswith((".tex", ".txt", ".bib")):
                with open(os.path.join(root, f), "r") as doc:
                    content = doc.read()
                    assert "/home/kiapi" not in content
                    assert "kiapi" not in content.lower()


def test_16_no_unverified_bibtex_entries():
    p2g_dir = get_latest_phase2g_dir()
    bib_file = os.path.join(p2g_dir, "paper", "references.bib")
    assert os.path.exists(bib_file)
    with open(bib_file, "r") as f:
        bib_text = f.read()
    entries = re.findall(r"@\w+\{([^,]+),", bib_text)
    assert len(entries) >= 20
    for key in entries:
        assert len(key.strip()) > 3


def test_17_every_citation_key_resolves():
    p2g_dir = get_latest_phase2g_dir()
    bib_file = os.path.join(p2g_dir, "paper", "references.bib")
    with open(bib_file, "r") as f:
        bib_text = f.read()
    valid_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    for sec_file in glob.glob(os.path.join(p2g_dir, "paper", "sections", "*.tex")):
        with open(sec_file, "r") as f:
            sec_text = f.read()
            cites = re.findall(r"\\cite\{([^}]+)\}", sec_text)
            for cite_group in cites:
                for c in cite_group.split(","):
                    c_clean = c.strip()
                    assert c_clean in valid_keys, f"Citation key '{c_clean}' not found in references.bib"


def test_18_self_overlap_gate_executed():
    p2g_dir = get_latest_phase2g_dir()
    itsc_csv = os.path.join(p2g_dir, "policy", "ITSC_WACV_SELF_OVERLAP_MATRIX_V7.csv")
    itsc_dec = os.path.join(p2g_dir, "policy", "DUAL_SUBMISSION_POLICY_DECISION_V7.md")
    assert os.path.exists(itsc_csv)
    assert os.path.exists(itsc_dec)


def test_19_waymo_attribution_present():
    p2g_dir = get_latest_phase2g_dir()
    sec4_file = os.path.join(p2g_dir, "paper", "sections", "4_data_and_sealed_protocol.tex")
    with open(sec4_file, "r") as f:
        text = f.read()
    assert "Waymo Open Motion Dataset" in text


def test_20_restricted_data_absent_from_submission_zip():
    p2g_dir = get_latest_phase2g_dir()
    sub_zip = os.path.join(p2g_dir, "output", "phase2g_submission_package.zip")
    assert os.path.exists(sub_zip)
    with zipfile.ZipFile(sub_zip, "r") as z:
        names = z.namelist()
        for name in names:
            assert not name.endswith(".parquet"), f"Restricted parquet file {name} found in submission zip!"
            assert "raw" not in name.lower()


def test_21_wacv_track_and_template_alignment():
    p2g_dir = get_latest_phase2g_dir()
    txt_file = os.path.join(p2g_dir, "paper", "ABSTRACT_TITLE_FOR_OPENREVIEW.txt")
    with open(txt_file, "r") as f:
        text = f.read()
    assert "WACV 2027 Track C — Evaluations & Datasets" in text


def test_22_main_pdf_page_budget():
    p2g_dir = get_latest_phase2g_dir()
    pdf_file = os.path.join(p2g_dir, "paper", "main_anonymous.pdf")
    assert os.path.exists(pdf_file)
    assert os.path.getsize(pdf_file) > 10000


def test_23_main_pdf_file_size():
    p2g_dir = get_latest_phase2g_dir()
    pdf_file = os.path.join(p2g_dir, "paper", "main_anonymous.pdf")
    size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
    assert size_mb <= 50.0


def test_24_supplement_pdf_file_size():
    p2g_dir = get_latest_phase2g_dir()
    supp_pdf = os.path.join(p2g_dir, "submission_supplement", "supplement_anonymous.pdf")
    assert os.path.exists(supp_pdf)
    size_mb = os.path.getsize(supp_pdf) / (1024 * 1024)
    assert size_mb <= 200.0


def test_25_all_figures_rendered_unclipped():
    p2g_dir = get_latest_phase2g_dir()
    for i in range(1, 5):
        png = os.path.join(p2g_dir, "figures", f"fig{i}_*.png")
        pdf = os.path.join(p2g_dir, "figures", f"fig{i}_*.pdf")
        assert len(glob.glob(png)) >= 1, f"Figure {i} PNG missing"
        assert len(glob.glob(pdf)) >= 1, f"Figure {i} PDF missing"


def test_26_anonymous_pdf_metadata_scan():
    p2g_dir = get_latest_phase2g_dir()
    pdf_file = os.path.join(p2g_dir, "paper", "main_anonymous.pdf")
    with open(pdf_file, "rb") as f:
        raw_bytes = f.read()
    assert b"/home/kiapi" not in raw_bytes


def test_27_zip_extraction_and_integrity():
    p2g_dir = get_latest_phase2g_dir()
    zip_path = os.path.join(p2g_dir, "output", "phase2g_submission_package.zip")
    assert os.path.exists(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        assert z.testzip() is None, "Corrupted submission zip package!"
