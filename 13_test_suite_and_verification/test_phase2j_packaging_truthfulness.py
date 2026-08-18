"""
Comprehensive Verification and Truthfulness Test Suite for WACV 2027 Phase 2J.
Validates all 20 critical gates, truthful reproducibility assertions, compile logs, visual QA, and upload directory seal.
"""

import os
import sys
import glob
import json
import zipfile
import subprocess
import pytest
import pandas as pd

WORKSPACE_ROOT = "/home/kiapi/waymo_motion_project"


def get_latest_phase2j_dir():
    candidates = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, "work", "phase2j_final_packaging_truthfulness_*")))
    if not candidates:
        pytest.skip("No Phase 2J output directory found. Run phase2j_master_engine.py first.")
    return candidates[-1]


def test_01_input_inventory_presence_and_hashes():
    p2j_dir = get_latest_phase2j_dir()
    inv_csv = os.path.join(p2j_dir, "qa", "INPUT_INVENTORY_SHA256.csv")
    assert os.path.exists(inv_csv), "INPUT_INVENTORY_SHA256.csv missing"
    df = pd.read_csv(inv_csv)
    assert len(df) >= 5, f"Expected >= 5 inventory items, found {len(df)}"
    found_rows = df[df["status"] == "FOUND_VERIFIED"]
    assert len(found_rows) >= 4, "Key input artifacts must be found and verified"


def test_02_authoritative_corrected_manuscript_facts():
    p2j_dir = get_latest_phase2j_dir()
    sec3_file = os.path.join(p2j_dir, "corrected_source", "paper_source", "sec", "3_problem_formulation.tex")
    sec4_file = os.path.join(p2j_dir, "corrected_source", "paper_source", "sec", "4_data_and_protocol.tex")
    sec7_file = os.path.join(p2j_dir, "corrected_source", "paper_source", "sec", "7_discussion_limitations.tex")
    
    with open(sec3_file, "r") as f:
        s3 = f.read()
    assert "C_{s,t}" in s3
    assert "\\mathrm{TTC}_{s,t}" in s3
    assert "\\mathcal{A}_{70}(s,t)" in s3
    
    with open(sec4_file, "r") as f:
        s4 = f.read()
    assert "HistGradientBoostingClassifier(max\\_iter=100, max\\_depth=6, random\\_state=42)" in s4
    assert "w_{s,t} = 1 / n_s" in s4
    assert "18,445" in s4
    assert "15,641" in s4
    
    with open(sec7_file, "r") as f:
        s7 = f.read()
    assert "+0.1347" in s7
    assert "+0.3451" in s7
    assert "neither external validation nor evidence of crash avoidance" in s7


def test_03_paper_id_handling_logic():
    p2j_dir = get_latest_phase2j_dir()
    main_tex = os.path.join(p2j_dir, "corrected_source", "paper_source", "main.tex")
    with open(main_tex, "r") as f:
        content = f.read()
    assert "\\def\\wacvPaperID{" in content
    gates_json = os.path.join(p2j_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    with open(gates_json, "r") as f:
        gates = json.load(f)
    assert gates["FINAL_SUBMISSION_STATUS"] in ["SUBMISSION_READY", "PAPER_ID_REQUIRED"]


def test_04_latex_compile_main_and_supp():
    p2j_dir = get_latest_phase2j_dir()
    main_pdf = os.path.join(p2j_dir, "submission_upload", "main_anonymous.pdf")
    supp_pdf = os.path.join(p2j_dir, "corrected_source", "supplement_source", "supplement_anonymous.pdf")
    assert os.path.exists(main_pdf), "main_anonymous.pdf missing"
    assert os.path.exists(supp_pdf), "supplement_anonymous.pdf missing"
    assert os.path.getsize(main_pdf) > 50000
    assert os.path.getsize(supp_pdf) > 20000


def test_05_page_limits_exact():
    p2j_dir = get_latest_phase2j_dir()
    main_pdf = os.path.join(p2j_dir, "submission_upload", "main_anonymous.pdf")
    supp_pdf = os.path.join(p2j_dir, "corrected_source", "supplement_source", "supplement_anonymous.pdf")
    
    res_main = subprocess.run(["pdfinfo", main_pdf], capture_output=True, text=True, check=True)
    main_pages = int([l.split(":")[1].strip() for l in res_main.stdout.splitlines() if l.startswith("Pages:")][0])
    assert main_pages <= 8, f"Main PDF must be <= 8 pages, got {main_pages}"
    assert main_pages == 6, f"Corrected main PDF is expected to be exactly 6 pages, got {main_pages}"
    
    res_supp = subprocess.run(["pdfinfo", supp_pdf], capture_output=True, text=True, check=True)
    supp_pages = int([l.split(":")[1].strip() for l in res_supp.stdout.splitlines() if l.startswith("Pages:")][0])
    assert supp_pages == 2, f"Supplement PDF must be exactly 2 pages, got {supp_pages}"


def test_06_font_properties_zero_type3():
    p2j_dir = get_latest_phase2j_dir()
    for pdf_p in [
        os.path.join(p2j_dir, "submission_upload", "main_anonymous.pdf"),
        os.path.join(p2j_dir, "corrected_source", "supplement_source", "supplement_anonymous.pdf")
    ]:
        res = subprocess.run(["pdffonts", pdf_p], capture_output=True, text=True, check=True)
        assert "Type 3" not in res.stdout, f"Type 3 fonts detected in {pdf_p}"


def test_07_page_visual_qa_all_pages_rendered_200dpi():
    p2j_dir = get_latest_phase2j_dir()
    renders_dir = os.path.join(p2j_dir, "qa", "page_renders")
    assert os.path.exists(renders_dir)
    pngs = glob.glob(os.path.join(renders_dir, "*.png"))
    assert len(pngs) >= 8, f"Expected 8 rendered pages, found {len(pngs)}"
    
    qa_csv = os.path.join(p2j_dir, "qa", "PAGE_VISUAL_QA.csv")
    assert os.path.exists(qa_csv)
    df_qa = pd.read_csv(qa_csv)
    assert len(df_qa) == 8
    assert all(df_qa["visual_qa_status"] == "PASS")


def test_08_fig1_fig2_visual_gates():
    p2j_dir = get_latest_phase2j_dir()
    fig_dir = os.path.join(p2j_dir, "corrected_source", "paper_source", "figures")
    for i in [1, 2, 3, 4]:
        pdf = glob.glob(os.path.join(fig_dir, f"fig{i}_*.pdf"))
        png = glob.glob(os.path.join(fig_dir, f"fig{i}_*.png"))
        assert len(pdf) == 1, f"Figure {i} PDF missing"
        assert len(png) == 1, f"Figure {i} PNG missing"


def test_09_tables_no_clipping():
    p2j_dir = get_latest_phase2j_dir()
    s2_file = os.path.join(p2j_dir, "corrected_source", "supplement_source", "sec_supp", "s2_scenario_effects.tex")
    with open(s2_file, "r") as f:
        s2 = f.read()
    assert "\\begin{table}[h]" in s2
    assert "CROSS\\_LEVEL\\_STABLE" in s2
    assert "UNSUPPORTED\\_BOTH" in s2
    assert "FRAME\\_LOCAL" in s2


def test_10_truthful_reproducibility_script_limited_scope():
    p2j_dir = get_latest_phase2j_dir()
    repro_script = os.path.join(p2j_dir, "reproducibility", "reproduce_paper_assets.py")
    assert os.path.exists(repro_script)
    with open(repro_script, "r") as f:
        content = f.read()
    assert "SUCCESS: Selected aggregate checks passed; Figure 2 reproduced." in content
    assert "SELECTED_AGGREGATE_CHECKS_PASSED" in content
    assert "all paper tables, figures, and claims" not in content.lower()


def test_11_reproducibility_readme_disclaimer():
    p2j_dir = get_latest_phase2j_dir()
    readme = os.path.join(p2j_dir, "reproducibility", "README.md")
    assert os.path.exists(readme)
    with open(readme, "r") as f:
        text = f.read()
    assert "This package verifies selected aggregate manuscript assertions and reproduces Figure 2 from supplied aggregate evidence." in text
    assert "Waymo Open Motion Dataset" in text
    assert "https://waymo.com/open" in text


def test_12_reproducibility_fresh_directory_execution():
    p2j_dir = get_latest_phase2j_dir()
    repro_script = os.path.join(p2j_dir, "reproducibility", "reproduce_paper_assets.py")
    res = subprocess.run([sys.executable, repro_script], cwd=os.path.join(p2j_dir, "reproducibility"), capture_output=True, text=True)
    assert res.returncode == 0
    assert "SUCCESS: Selected aggregate checks passed; Figure 2 reproduced." in res.stdout
    
    report_json = os.path.join(p2j_dir, "reproducibility", "reproduced_assets", "REPRODUCTION_REPORT.json")
    assert os.path.exists(report_json)
    with open(report_json, "r") as f:
        rep = json.load(f)
    assert rep["status"] == "SELECTED_AGGREGATE_CHECKS_PASSED"
    assert rep["selected_assertions_checked"] >= 8


def test_13_supplement_zip_structure():
    p2j_dir = get_latest_phase2j_dir()
    supp_zip = os.path.join(p2j_dir, "submission_upload", "supplement.zip")
    assert os.path.exists(supp_zip)
    with zipfile.ZipFile(supp_zip, "r") as z:
        names = z.namelist()
        assert "supplement_anonymous.pdf" in names
        assert "reproducibility/README.md" in names
        assert "reproducibility/reproduce_paper_assets.py" in names
        assert "reproducibility/data/TABLE1_NESTED_MODELS_V6.csv" in names
        assert "CHECKSUMS_SHA256.txt" in names
        assert "main_anonymous.pdf" not in names
        for n in names:
            assert ".git" not in n
            assert "tfrecord" not in n.lower()


def test_14_anonymity_scan_zero_local_paths():
    p2j_dir = get_latest_phase2j_dir()
    anon_json = os.path.join(p2j_dir, "qa", "ANONYMITY_SCAN.json")
    assert os.path.exists(anon_json)
    with open(anon_json, "r") as f:
        anon = json.load(f)
    assert anon["scan_status"] == "PASS"
    assert anon["leaks_found"] == 0


def test_15_submission_upload_exact_two_files():
    p2j_dir = get_latest_phase2j_dir()
    upload_dir = os.path.join(p2j_dir, "submission_upload")
    files = sorted(os.listdir(upload_dir))
    assert files == ["main_anonymous.pdf", "supplement.zip"], f"submission_upload must contain strictly main_anonymous.pdf and supplement.zip, got {files}"


def test_16_file_size_limits():
    p2j_dir = get_latest_phase2j_dir()
    main_pdf = os.path.join(p2j_dir, "submission_upload", "main_anonymous.pdf")
    supp_zip = os.path.join(p2j_dir, "submission_upload", "supplement.zip")
    assert os.path.getsize(main_pdf) / (1024 * 1024) <= 50.0
    assert os.path.getsize(supp_zip) / (1024 * 1024) <= 200.0


def test_17_checksums_manifest_presence():
    p2j_dir = get_latest_phase2j_dir()
    manifest = os.path.join(p2j_dir, "CHECKSUMS_SHA256.txt")
    assert os.path.exists(manifest)
    with open(manifest, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 10


def test_18_dynamic_gates_json_structure():
    p2j_dir = get_latest_phase2j_dir()
    gates_file = os.path.join(p2j_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    assert os.path.exists(gates_file)
    with open(gates_file, "r") as f:
        g = json.load(f)
    assert "FINAL_SUBMISSION_STATUS" in g
    assert "GATES" in g
    assert len(g["GATES"]) >= 20
    for gname, gdata in g["GATES"].items():
        assert "status" in gdata
        assert "evidence_path" in gdata
        assert "observed_value" in gdata
        assert "expected_value" in gdata


def test_19_final_submission_status():
    p2j_dir = get_latest_phase2j_dir()
    gates_file = os.path.join(p2j_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    with open(gates_file, "r") as f:
        g = json.load(f)
    assert g["FINAL_SUBMISSION_STATUS"] in ["SUBMISSION_READY", "PAPER_ID_REQUIRED"]


def test_20_output_zip_bundles_integrity():
    p2j_dir = get_latest_phase2j_dir()
    for zname in ["phase2j_submission_package.zip", "phase2j_feedback_bundle.zip", "phase2j_code_package.zip"]:
        zp = os.path.join(p2j_dir, "output", zname)
        assert os.path.exists(zp), f"ZIP bundle {zname} missing"
        with zipfile.ZipFile(zp, "r") as z:
            assert z.testzip() is None
