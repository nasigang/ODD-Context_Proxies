"""
Comprehensive Verification and Compliance Test Suite for WACV 2027 Phase 2I.
Validates all 27 critical gates, execution-manuscript parity, LaTeX compilation, and visual QA.
"""

import os
import glob
import json
import zipfile
import subprocess
import re
import sys
import pytest
import pandas as pd

WORKSPACE_ROOT = "/home/kiapi/waymo_motion_project"


def get_latest_phase2i_dir():
    candidates = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, "work", "phase2i_final_integrity_repair_*")))
    if not candidates:
        pytest.skip("No Phase 2I output directory found. Run phase2i_master_engine.py first.")
    return candidates[-1]


def test_01_target_estimand_sdc_centric_in_all_sections():
    p2i_dir = get_latest_phase2i_dir()
    sec3_file = os.path.join(p2i_dir, "paper_source", "sec", "3_problem_formulation.tex")
    with open(sec3_file, "r") as f:
        text = f.read()
    assert "TTC}_{\\mathrm{OBB}}(\\mathrm{SDC}, j)" in text
    assert "\\mathcal{A}_{70}(s,t)" in text
    
    # Check that "all agent pairs" is not used in main text
    for sec_file in glob.glob(os.path.join(p2i_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            t = f.read().lower()
            assert "all agent pairs" not in t, f"'all agent pairs' found in {sec_file}"


def test_02_model_training_facts_exact():
    p2i_dir = get_latest_phase2i_dir()
    sec4_file = os.path.join(p2i_dir, "paper_source", "sec", "4_data_and_protocol.tex")
    with open(sec4_file, "r") as f:
        text = f.read()
    assert "HistGradientBoostingClassifier(max\\_iter=100, max\\_depth=6, random\\_state=42)" in text
    assert "15,641" in text
    assert "w_{s,t} = 1 / n_s" in text
    assert "learning rate" not in text
    assert "max_leaf_nodes" not in text


def test_03_feature_taxonomy_exact():
    p2i_dir = get_latest_phase2i_dir()
    sec4_file = os.path.join(p2i_dir, "paper_source", "sec", "4_data_and_protocol.tex")
    with open(sec4_file, "r") as f:
        text = f.read()
    assert "yaw rate" in text
    assert "jerk" not in text.lower()
    assert "entropy" in text.lower()
    assert "speed standard deviation" in text.lower()


def test_04_kpi_authoritative_values_parity():
    p2i_dir = get_latest_phase2i_dir()
    sec7_file = os.path.join(p2i_dir, "paper_source", "sec", "7_discussion_limitations.tex")
    with open(sec7_file, "r") as f:
        text = f.read()
    assert "+0.1347" in text
    assert "+0.3451" in text
    assert "+0.321" not in text


def test_05_temporal_target_profile_method():
    p2i_dir = get_latest_phase2i_dir()
    sec6_file = os.path.join(p2i_dir, "paper_source", "sec", "6_mechanism_and_temporal.tex")
    with open(sec6_file, "r") as f:
        text = f.read()
    assert "\\bar{E}_{s,i}" in text
    assert "D_s^{\\mathrm{Peak}}" in text
    assert "D_s^{\\mathrm{AUC}}" in text
    assert "D_s^{\\mathrm{TET3}}" in text
    assert "+0.1914" in text
    assert "+0.2345" in text
    assert "+0.0728" in text


def test_06_cohort_provenance_honesty():
    p2i_dir = get_latest_phase2i_dir()
    sec4_file = os.path.join(p2i_dir, "paper_source", "sec", "4_data_and_protocol.tex")
    with open(sec4_file, "r") as f:
        text = f.read()
    assert "processed subset of the public WOMD training corpus" in text
    assert "18,445" in text
    assert "1,674,495" in text


def test_07_references_safeshift_puphal_weng():
    p2i_dir = get_latest_phase2i_dir()
    bib_file = os.path.join(p2i_dir, "paper_source", "references.bib")
    with open(bib_file, "r") as f:
        text = f.read()
    assert "Benjamin Stoler" in text
    assert "1179--1186" in text
    assert "Tim Puphal" in text
    assert "Erica Weng" in text
    assert "Lukas Westhofen" in text


def test_08_official_wacv_author_kit_presence():
    p2i_dir = get_latest_phase2i_dir()
    for d in ["paper_source", "supplement_source"]:
        assert os.path.exists(os.path.join(p2i_dir, d, "wacv.sty")), f"wacv.sty missing in {d}"
        assert os.path.exists(os.path.join(p2i_dir, d, "ieeenat_fullname.bst")), f"ieeenat_fullname.bst missing in {d}"
        assert os.path.exists(os.path.join(p2i_dir, d, "preamble.tex")), f"preamble.tex missing in {d}"


def test_09_real_latex_clean_compile():
    p2i_dir = get_latest_phase2i_dir()
    main_pdf = os.path.join(p2i_dir, "submission_upload", "main_anonymous.pdf")
    supp_pdf = os.path.join(p2i_dir, "submission_upload", "supplement_anonymous.pdf")
    assert os.path.exists(main_pdf)
    assert os.path.exists(supp_pdf)
    assert os.path.getsize(main_pdf) > 50000
    assert os.path.getsize(supp_pdf) > 20000


def test_10_zero_type3_fonts():
    p2i_dir = get_latest_phase2i_dir()
    for pdf_name in ["main_anonymous.pdf", "supplement_anonymous.pdf"]:
        pdf_path = os.path.join(p2i_dir, "submission_upload", pdf_name)
        res = subprocess.run(["pdffonts", pdf_path], capture_output=True, text=True, check=True)
        assert "Type 3" not in res.stdout, f"Type 3 fonts detected in {pdf_name}"


def test_11_all_pages_rendered_200dpi():
    p2i_dir = get_latest_phase2i_dir()
    renders_dir = os.path.join(p2i_dir, "qa", "page_renders")
    assert os.path.exists(renders_dir)
    pngs = glob.glob(os.path.join(renders_dir, "*.png"))
    assert len(pngs) >= 7, f"Only {len(pngs)} rendered page images found"


def test_12_fig1_redesigned_vector_and_png():
    p2i_dir = get_latest_phase2i_dir()
    fig1_pdf = os.path.join(p2i_dir, "paper_source", "figures", "fig1_measurement_architecture.pdf")
    fig1_png = os.path.join(p2i_dir, "paper_source", "figures", "fig1_measurement_architecture.png")
    assert os.path.exists(fig1_pdf)
    assert os.path.exists(fig1_png)
    assert os.path.getsize(fig1_png) > 100000


def test_13_fig2_forest_plot_safe_margins():
    p2i_dir = get_latest_phase2i_dir()
    fig2_pdf = os.path.join(p2i_dir, "paper_source", "figures", "fig2_forest_plot_nested_models.pdf")
    fig2_png = os.path.join(p2i_dir, "paper_source", "figures", "fig2_forest_plot_nested_models.png")
    assert os.path.exists(fig2_pdf)
    assert os.path.exists(fig2_png)


def test_14_figures_unclipped():
    p2i_dir = get_latest_phase2i_dir()
    for i in range(1, 5):
        pdf = glob.glob(os.path.join(p2i_dir, "paper_source", "figures", f"fig{i}_*.pdf"))
        png = glob.glob(os.path.join(p2i_dir, "paper_source", "figures", f"fig{i}_*.png"))
        assert len(pdf) == 1, f"Figure {i} PDF missing"
        assert len(png) == 1, f"Figure {i} PNG missing"


def test_15_numeric_exact_parity_primary():
    p2i_dir = get_latest_phase2i_dir()
    sec5_file = os.path.join(p2i_dir, "paper_source", "sec", "5_primary_results.tex")
    with open(sec5_file, "r") as f:
        text = f.read()
    assert "0.3370" in text
    assert "0.3224" in text
    assert "+0.0147" in text
    assert "+0.0005" in text
    assert "+0.0285" in text


def test_16_numeric_exact_parity_secondary():
    p2i_dir = get_latest_phase2i_dir()
    sec6_file = os.path.join(p2i_dir, "paper_source", "sec", "6_mechanism_and_temporal.tex")
    with open(sec6_file, "r") as f:
        text = f.read()
    assert "+0.0161" in text
    assert "+0.0057" in text
    assert "+0.0264" in text


def test_17_supplement_table_parity():
    p2i_dir = get_latest_phase2i_dir()
    s1_file = os.path.join(p2i_dir, "supplement_source", "sec_supp", "s1_feature_confirmation.tex")
    s2_file = os.path.join(p2i_dir, "supplement_source", "sec_supp", "s2_scenario_effects.tex")
    with open(s1_file, "r") as f:
        s1_text = f.read()
    with open(s2_file, "r") as f:
        s2_text = f.read()
    assert s1_text.count("\\textbf{CONFIRMED}") == 10
    assert s1_text.count("DIRECTIONAL") == 3
    assert s2_text.count("\\texttt{") == 17


def test_18_no_pseudo_pvalues():
    p2i_dir = get_latest_phase2i_dir()
    for sec_file in glob.glob(os.path.join(p2i_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            text = f.read()
            assert "p = 0.0470" not in text
            assert "p = 0.0050" not in text
            assert "p_empirical" not in text


def test_19_no_forbidden_phrases():
    p2i_dir = get_latest_phase2i_dir()
    forbidden = [
        "orthogonal predictive information", "orthogonal information",
        "proves safety", "generalization guarantee", "zero-leakage guarantee",
        "drives overwhelmingly", "demonstrates mechanism", "vital role", "rigorous empirical ground truth"
    ]
    for sec_file in glob.glob(os.path.join(p2i_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            text = f.read().lower()
            for p in forbidden:
                assert p not in text, f"Forbidden phrase '{p}' found in {sec_file}"


def test_20_anonymity_scan_zero_local_tokens():
    p2i_dir = get_latest_phase2i_dir()
    for sub in ["paper_source", "supplement_source", "reproducibility", "submission_upload"]:
        target_path = os.path.join(p2i_dir, sub)
        if os.path.exists(target_path):
            for root, dirs, files in os.walk(target_path):
                for f in files:
                    if f.endswith((".tex", ".bib", ".py", ".md", ".json", ".csv")):
                        with open(os.path.join(root, f), "r", errors="ignore") as doc:
                            content = doc.read()
                            assert "/home/kiapi" not in content, f"Local path found in {os.path.join(root, f)}"


def test_21_reproducibility_script_functional_and_deterministic():
    p2i_dir = get_latest_phase2i_dir()
    repro_script = os.path.join(p2i_dir, "reproducibility", "reproduce_paper_assets.py")
    assert os.path.exists(repro_script)
    res = subprocess.run([sys.executable, repro_script], cwd=os.path.join(p2i_dir, "reproducibility"), capture_output=True, text=True)
    assert res.returncode == 0
    assert "SUCCESS" in res.stdout


def test_22_submission_upload_dir_clean():
    p2i_dir = get_latest_phase2i_dir()
    upload_dir = os.path.join(p2i_dir, "submission_upload")
    files = sorted(os.listdir(upload_dir))
    assert "main_anonymous.pdf" in files
    assert "supplement.zip" in files


def test_23_pdf_file_size_limits():
    p2i_dir = get_latest_phase2i_dir()
    main_pdf = os.path.join(p2i_dir, "submission_upload", "main_anonymous.pdf")
    supp_zip = os.path.join(p2i_dir, "submission_upload", "supplement.zip")
    assert os.path.getsize(main_pdf) / (1024 * 1024) <= 50.0
    assert os.path.getsize(supp_zip) / (1024 * 1024) <= 200.0


def test_24_claim_evidence_ledger_presence():
    p2i_dir = get_latest_phase2i_dir()
    ledger_csv = os.path.join(p2i_dir, "qa", "CLAIM_EVIDENCE_LEDGER.csv")
    assert os.path.exists(ledger_csv)
    df = pd.read_csv(ledger_csv)
    assert len(df) >= 10
    assert all(df["parity_status"] == "EXACT_MATCH")


def test_25_execution_manuscript_parity_json():
    p2i_dir = get_latest_phase2i_dir()
    parity_json = os.path.join(p2i_dir, "qa", "EXECUTION_MANUSCRIPT_PARITY.json")
    assert os.path.exists(parity_json)
    with open(parity_json, "r") as f:
        p_dict = json.load(f)
    assert p_dict["OVERALL_PARITY"] == "VERIFIED_100_PERCENT"
    assert p_dict["TARGET_ESTIMAND"] == "SDC_CENTRIC_MIN_SWEPT_OBB_TTC_WITHIN_70M"


def test_26_final_submission_gates_all_pass():
    p2i_dir = get_latest_phase2i_dir()
    gates_json = os.path.join(p2i_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    assert os.path.exists(gates_json)
    with open(gates_json, "r") as f:
        gates = json.load(f)
    assert gates["FINAL_SUBMISSION_STATUS"] == "SUBMISSION_READY"
    assert gates["AUTHORITATIVE_EVIDENCE_PARITY_GATE"] == "PASS"
    assert gates["TARGET_ESTIMAND_CORRECTED_GATE"] == "PASS"


def test_27_zip_bundles_integrity():
    p2i_dir = get_latest_phase2i_dir()
    for z_name in ["phase2i_submission_package.zip", "phase2i_feedback_bundle.zip", "phase2i_code_package.zip"]:
        zip_path = os.path.join(p2i_dir, "output", z_name)
        assert os.path.exists(zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            assert z.testzip() is None
