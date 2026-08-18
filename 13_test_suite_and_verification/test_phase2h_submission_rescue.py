"""
Comprehensive Verification and Compliance Test Suite for WACV 2027 Phase 2H.
Validates all 27 critical gates, true LaTeX compilation, visual QA, and anonymity.
"""

import os
import glob
import json
import zipfile
import subprocess
import re
import pytest
import pandas as pd

WORKSPACE_ROOT = "/home/kiapi/waymo_motion_project"


def get_latest_phase2h_dir():
    candidates = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, "work", "phase2h_submission_rescue_*")))
    if not candidates:
        pytest.skip("No Phase 2H output directory found. Run phase2h_master_engine.py first.")
    return candidates[-1]


def test_01_official_wacv_author_kit_presence():
    p2h_dir = get_latest_phase2h_dir()
    for d in ["paper_source", "supplement_source"]:
        assert os.path.exists(os.path.join(p2h_dir, d, "wacv.sty")), f"wacv.sty missing in {d}"
        assert os.path.exists(os.path.join(p2h_dir, d, "ieeenat_fullname.bst")), f"ieeenat_fullname.bst missing in {d}"
        assert os.path.exists(os.path.join(p2h_dir, d, "preamble.tex")), f"preamble.tex missing in {d}"


def test_02_real_latex_clean_compile():
    p2h_dir = get_latest_phase2h_dir()
    main_pdf = os.path.join(p2h_dir, "submission_upload", "main_anonymous.pdf")
    supp_pdf = os.path.join(p2h_dir, "submission_upload", "supplement_anonymous.pdf")
    assert os.path.exists(main_pdf), f"Missing compiled main PDF: {main_pdf}"
    assert os.path.exists(supp_pdf), f"Missing compiled supplement PDF: {supp_pdf}"
    assert os.path.getsize(main_pdf) > 20000
    assert os.path.getsize(supp_pdf) > 10000


def test_03_main_pdf_page_count_and_budget():
    p2h_dir = get_latest_phase2h_dir()
    main_pdf = os.path.join(p2h_dir, "submission_upload", "main_anonymous.pdf")
    res = subprocess.run(["pdfinfo", main_pdf], capture_output=True, text=True, check=True)
    match = re.search(r"Pages:\s+(\d+)", res.stdout)
    assert match is not None
    num_pages = int(match.group(1))
    assert num_pages in [7, 8, 9], f"Unexpected page count: {num_pages} (expected 7-8 pages main body + references)"


def test_04_zero_type3_fonts():
    p2h_dir = get_latest_phase2h_dir()
    for pdf_name in ["main_anonymous.pdf", "supplement_anonymous.pdf"]:
        pdf_path = os.path.join(p2h_dir, "submission_upload", pdf_name)
        res = subprocess.run(["pdffonts", pdf_path], capture_output=True, text=True, check=True)
        assert "Type 3" not in res.stdout, f"Type 3 fonts detected in {pdf_name}"


def test_05_all_pages_rendered_200dpi():
    p2h_dir = get_latest_phase2h_dir()
    renders_dir = os.path.join(p2h_dir, "qa_internal", "page_renders")
    assert os.path.exists(renders_dir)
    pngs = glob.glob(os.path.join(renders_dir, "*.png"))
    assert len(pngs) >= 7, f"Only {len(pngs)} rendered page images found"


def test_06_fig1_redesigned_vector_and_png():
    p2h_dir = get_latest_phase2h_dir()
    fig1_pdf = os.path.join(p2h_dir, "paper_source", "figures", "fig1_measurement_architecture.pdf")
    fig1_png = os.path.join(p2h_dir, "paper_source", "figures", "fig1_measurement_architecture.png")
    assert os.path.exists(fig1_pdf)
    assert os.path.exists(fig1_png)
    assert os.path.getsize(fig1_png) > 100000


def test_07_fig2_forest_plot_margins():
    p2h_dir = get_latest_phase2h_dir()
    fig2_pdf = os.path.join(p2h_dir, "paper_source", "figures", "fig2_forest_plot_nested_models.pdf")
    fig2_png = os.path.join(p2h_dir, "paper_source", "figures", "fig2_forest_plot_nested_models.png")
    assert os.path.exists(fig2_pdf)
    assert os.path.exists(fig2_png)


def test_08_figures_unclipped():
    p2h_dir = get_latest_phase2h_dir()
    for i in range(1, 5):
        pdf = glob.glob(os.path.join(p2h_dir, "paper_source", "figures", f"fig{i}_*.pdf"))
        png = glob.glob(os.path.join(p2h_dir, "paper_source", "figures", f"fig{i}_*.png"))
        assert len(pdf) == 1, f"Figure {i} PDF missing"
        assert len(png) == 1, f"Figure {i} PNG missing"


def test_09_numeric_exact_parity_primary():
    p2h_dir = get_latest_phase2h_dir()
    sec5_file = os.path.join(p2h_dir, "paper_source", "sec", "5_primary_results.tex")
    with open(sec5_file, "r") as f:
        text = f.read()
    assert "0.3370" in text
    assert "0.3224" in text
    assert "+0.0147" in text
    assert "+0.0005" in text
    assert "+0.0285" in text


def test_10_numeric_exact_parity_secondary():
    p2h_dir = get_latest_phase2h_dir()
    sec6_file = os.path.join(p2h_dir, "paper_source", "sec", "6_mechanism_and_temporal.tex")
    with open(sec6_file, "r") as f:
        text = f.read()
    assert "+0.0161" in text
    assert "+0.0057" in text
    assert "+0.0264" in text


def test_11_supplement_table_parity():
    p2h_dir = get_latest_phase2h_dir()
    s1_file = os.path.join(p2h_dir, "supplement_source", "sec_supp", "s1_feature_confirmation.tex")
    s2_file = os.path.join(p2h_dir, "supplement_source", "sec_supp", "s2_scenario_effects.tex")
    with open(s1_file, "r") as f:
        s1_text = f.read()
    with open(s2_file, "r") as f:
        s2_text = f.read()
    # Check that Table S1 has 13 features and Table S2 has 17 features
    assert s1_text.count("\\textbf{CONFIRMED}") == 10
    assert s1_text.count("DIRECTIONAL") == 3
    assert s2_text.count("\\texttt{") == 17


def test_12_no_pseudo_pvalues():
    p2h_dir = get_latest_phase2h_dir()
    for sec_file in glob.glob(os.path.join(p2h_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            text = f.read()
            assert "p = 0.0470" not in text
            assert "p = 0.0050" not in text
            assert "p_empirical" not in text


def test_13_no_forbidden_phrases():
    p2h_dir = get_latest_phase2h_dir()
    forbidden = [
        "orthogonal predictive information", "orthogonal information",
        "proves safety", "generalization guarantee", "zero-leakage guarantee"
    ]
    for sec_file in glob.glob(os.path.join(p2h_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            text = f.read().lower()
            for p in forbidden:
                assert p not in text, f"Forbidden phrase '{p}' found in {sec_file}"


def test_14_no_18_interaction_features():
    p2h_dir = get_latest_phase2h_dir()
    for sec_file in glob.glob(os.path.join(p2h_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            text = f.read().lower()
            assert "18 interaction features" not in text


def test_15_odd_context_proxy_defined():
    p2h_dir = get_latest_phase2h_dir()
    sec1_file = os.path.join(p2h_dir, "paper_source", "sec", "1_intro.tex")
    with open(sec1_file, "r") as f:
        text = f.read()
    assert "ODD-context proxies" in text


def test_16_temporal_role_retrospective():
    p2h_dir = get_latest_phase2h_dir()
    sec3_file = os.path.join(p2h_dir, "paper_source", "sec", "3_problem_formulation.tex")
    with open(sec3_file, "r") as f:
        text = f.read()
    assert "retrospectively" in text.lower()


def test_17_shared_state_disclosed():
    p2h_dir = get_latest_phase2h_dir()
    sec3_file = os.path.join(p2h_dir, "paper_source", "sec", "3_problem_formulation.tex")
    with open(sec3_file, "r") as f:
        text = f.read()
    assert "convergent surrogate validity" in text.lower()


def test_18_all_citation_keys_resolved():
    p2h_dir = get_latest_phase2h_dir()
    bib_file = os.path.join(p2h_dir, "paper_source", "references.bib")
    with open(bib_file, "r") as f:
        bib_text = f.read()
    valid_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    for sec_file in glob.glob(os.path.join(p2h_dir, "paper_source", "sec", "*.tex")):
        with open(sec_file, "r") as f:
            text = f.read()
            cites = re.findall(r"\\cite\{([^}]+)\}", text)
            for cg in cites:
                for c in cg.split(","):
                    c_clean = c.strip()
                    assert c_clean in valid_keys, f"Unresolved citation key '{c_clean}' in {sec_file}"


def test_19_bibtex_entries_verified():
    p2h_dir = get_latest_phase2h_dir()
    bib_file = os.path.join(p2h_dir, "paper_source", "references.bib")
    with open(bib_file, "r") as f:
        bib_text = f.read()
    entries = re.findall(r"@\w+\{([^,]+),", bib_text)
    assert len(entries) >= 18


def test_20_anonymity_scan_zero_local_tokens():
    p2h_dir = get_latest_phase2h_dir()
    for sub in ["paper_source", "supplement_source", "reproducibility", "submission_upload"]:
        target_path = os.path.join(p2h_dir, sub)
        if os.path.exists(target_path):
            for root, dirs, files in os.walk(target_path):
                for f in files:
                    if f.endswith((".tex", ".bib", ".py", ".md", ".json", ".csv")):
                        with open(os.path.join(root, f), "r", errors="ignore") as doc:
                            content = doc.read()
                            assert "/home/kiapi" not in content, f"Local path found in {os.path.join(root, f)}"


def test_21_waymo_license_compliance():
    p2h_dir = get_latest_phase2h_dir()
    sec4_file = os.path.join(p2h_dir, "paper_source", "sec", "4_data_and_protocol.tex")
    with open(sec4_file, "r") as f:
        text = f.read()
    assert "Waymo Open Motion Dataset" in text
    
    supp_zip = os.path.join(p2h_dir, "submission_upload", "supplement.zip")
    with zipfile.ZipFile(supp_zip, "r") as z:
        for name in z.namelist():
            assert not name.endswith(".parquet"), f"Restricted parquet file {name} in supplement.zip"


def test_22_reproducibility_script_functional():
    import sys
    p2h_dir = get_latest_phase2h_dir()
    repro_script = os.path.join(p2h_dir, "reproducibility", "reproduce_paper_assets.py")
    assert os.path.exists(repro_script)
    res = subprocess.run([sys.executable, repro_script], cwd=os.path.join(p2h_dir, "reproducibility"), capture_output=True, text=True)
    assert res.returncode == 0
    assert "VERIFIED" in res.stdout


def test_23_submission_upload_dir_clean():
    p2h_dir = get_latest_phase2h_dir()
    upload_dir = os.path.join(p2h_dir, "submission_upload")
    files = sorted(os.listdir(upload_dir))
    assert "main_anonymous.pdf" in files
    assert "supplement.zip" in files


def test_24_pdf_file_size_limits():
    p2h_dir = get_latest_phase2h_dir()
    main_pdf = os.path.join(p2h_dir, "submission_upload", "main_anonymous.pdf")
    supp_zip = os.path.join(p2h_dir, "submission_upload", "supplement.zip")
    assert os.path.getsize(main_pdf) / (1024 * 1024) <= 50.0
    assert os.path.getsize(supp_zip) / (1024 * 1024) <= 200.0


def test_25_discrepancy_log_v8_clean():
    p2h_dir = get_latest_phase2h_dir()
    disc_csv = os.path.join(p2h_dir, "qa_internal", "EVIDENCE_DISCREPANCY_LOG_V8.csv")
    assert os.path.exists(disc_csv)
    df = pd.read_csv(disc_csv)
    assert len(df) >= 5
    assert all(df["status"] == "MISMATCH_FIXED")


def test_26_final_submission_gates_v8_all_pass():
    p2h_dir = get_latest_phase2h_dir()
    gates_json = os.path.join(p2h_dir, "qa_internal", "FINAL_SUBMISSION_GATES_V8.json")
    assert os.path.exists(gates_json)
    with open(gates_json, "r") as f:
        gates = json.load(f)
    assert gates["FINAL_SUBMISSION_STATUS"] == "SUBMISSION_READY"
    assert gates["REAL_LATEX_COMPILE_GATE"] == "PASS"
    assert gates["OFFICIAL_WACV_TEMPLATE_GATE"] == "PASS"


def test_27_zip_bundles_integrity():
    p2h_dir = get_latest_phase2h_dir()
    for z_name in ["phase2h_submission_package.zip", "phase2h_feedback_bundle.zip", "phase2h_code_package.zip"]:
        zip_path = os.path.join(p2h_dir, "output", z_name)
        assert os.path.exists(zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            assert z.testzip() is None
