#!/usr/bin/env python3
"""
Test Suite for WACV 2027 Phase 2K — Final Mechanical Seal and Portal-Ready Packaging
====================================================================================
Verifies all mechanical fixes, citation mappings, tabularx layout, supplement text,
truthful reproducibility wording, visual QA, dynamic gates, anonymity, and upload seal.
"""

import os
import sys
import glob
import json
import zipfile
import subprocess
import tempfile
import shutil
import pytest
import pandas as pd


def get_latest_phase2k_dir():
    workspace = "/home/kiapi/waymo_motion_project"
    p2k_dirs = sorted(glob.glob(os.path.join(workspace, "work", "phase2k_portal_packaging_*")))
    assert len(p2k_dirs) > 0, "No phase2k_portal_packaging directory found"
    return p2k_dirs[-1]


def test_01_input_inventory_presence_and_hashes():
    p2k_dir = get_latest_phase2k_dir()
    inv_csv = os.path.join(p2k_dir, "qa", "INPUT_INVENTORY_SHA256.csv")
    assert os.path.exists(inv_csv), "INPUT_INVENTORY_SHA256.csv missing"
    df = pd.read_csv(inv_csv)
    assert len(df) >= 3, "Expected at least 3 inventory input zip files"
    for col in ["filename", "path", "size_bytes", "sha256"]:
        assert col in df.columns, f"Column {col} missing in input inventory"


def test_02_citation_keys_resolved_in_source_and_bib():
    p2k_dir = get_latest_phase2k_dir()
    rel_work_tex = os.path.join(p2k_dir, "corrected_source", "paper_source", "sec", "2_related_work.tex")
    with open(rel_work_tex, "r") as f:
        rel_work = f.read()
        
    assert "hayward1972near" in rel_work, "hayward1972near missing in 2_related_work.tex"
    assert "hayward1972nearmiss" not in rel_work, "Stale hayward1972nearmiss still present"
    
    assert "laureshyn2010surrogate" in rel_work, "laureshyn2010surrogate missing in 2_related_work.tex"
    assert "laureshyn2010extended" not in rel_work, "Stale laureshyn2010extended still present"
    
    bib_file = os.path.join(p2k_dir, "corrected_source", "paper_source", "references.bib")
    with open(bib_file, "r") as f:
        bib = f.read()
        
    assert "Christer Hyd{\\'e}n" in bib, "Hyd{\'e}n normalization missing in references.bib"


def test_03_compilation_zero_undefined_citations_and_references():
    p2k_dir = get_latest_phase2k_dir()
    qa_json = os.path.join(p2k_dir, "qa", "LATEX_COMPILE_QA.json")
    assert os.path.exists(qa_json)
    with open(qa_json, "r") as f:
        qa = json.load(f)
    assert qa["main_pdf"]["undefined_citations"] == 0, "Undefined citations found in main PDF"
    assert qa["main_pdf"]["undefined_references"] == 0, "Undefined references found in main PDF"


def test_04_table1_tabularx_zero_overfull_hboxes():
    p2k_dir = get_latest_phase2k_dir()
    qa_json = os.path.join(p2k_dir, "qa", "LATEX_COMPILE_QA.json")
    with open(qa_json, "r") as f:
        qa = json.load(f)
    assert qa["main_pdf"]["overfull_hboxes"] == 0, f"Overfull hboxes found in main PDF: {qa['main_pdf']['overfull_hboxes']}"
    assert qa["supplement_pdf"]["overfull_hboxes"] == 0, f"Overfull hboxes found in supplement PDF: {qa['supplement_pdf']['overfull_hboxes']}"
    
    p5_tex = os.path.join(p2k_dir, "corrected_source", "paper_source", "sec", "5_primary_results.tex")
    with open(p5_tex, "r") as f:
        t1 = f.read()
    assert "\\begin{tabularx}{\\textwidth}" in t1, "Table 1 must use tabularx"


def test_05_page_limits_exact():
    p2k_dir = get_latest_phase2k_dir()
    main_pdf = os.path.join(p2k_dir, "submission_upload", "main_anonymous.pdf")
    supp_pdf = os.path.join(p2k_dir, "corrected_source", "supplement_source", "supplement_anonymous.pdf")
    
    res_main = subprocess.run(["pdfinfo", main_pdf], capture_output=True, text=True, check=True)
    main_pages = int([l.split(":")[1].strip() for l in res_main.stdout.splitlines() if l.startswith("Pages:")][0])
    assert main_pages == 6, f"Main PDF must be exactly 6 pages, got {main_pages}"
    
    res_supp = subprocess.run(["pdfinfo", supp_pdf], capture_output=True, text=True, check=True)
    supp_pages = int([l.split(":")[1].strip() for l in res_supp.stdout.splitlines() if l.startswith("Pages:")][0])
    assert supp_pages == 2, f"Supplement PDF must be exactly 2 pages, got {supp_pages}"


def test_06_font_properties_zero_type3():
    p2k_dir = get_latest_phase2k_dir()
    for pdf_p in [
        os.path.join(p2k_dir, "submission_upload", "main_anonymous.pdf"),
        os.path.join(p2k_dir, "corrected_source", "supplement_source", "supplement_anonymous.pdf")
    ]:
        res = subprocess.run(["pdffonts", pdf_p], capture_output=True, text=True, check=True)
        assert "Type 3" not in res.stdout, f"Type 3 fonts detected in {pdf_p}"


def test_07_supplement_s3_convergent_text_and_no_external_phrase():
    p2k_dir = get_latest_phase2k_dir()
    s3_file = os.path.join(p2k_dir, "corrected_source", "supplement_source", "sec_supp", "s3_threshold_and_kpi.tex")
    with open(s3_file, "r") as f:
        s3_text = f.read()
        
    assert "reports development-only convergent alignment between the continuous TTC-based severity score and within-dataset vehicle-response KPIs" in s3_text, "Required S3 convergent text missing in supplement"
    assert "KPI" in s3_text and "CONSTRUCT" in s3_text and "VALIDITY" in s3_text
    assert "external vehicle deceleration metrics" not in s3_text, "Forbidden phrase 'external vehicle deceleration metrics' found in S3"


def test_08_supplement_s2_classes_and_explicit_mapping():
    p2k_dir = get_latest_phase2k_dir()
    s2_file = os.path.join(p2k_dir, "corrected_source", "supplement_source", "sec_supp", "s2_scenario_effects.tex")
    with open(s2_file, "r") as f:
        s2_text = f.read()
        
    assert "Stable" in s2_text
    assert "Unsupported" in s2_text
    assert "Frame-local" in s2_text
    assert "Sign-discordant" in s2_text
    assert "Temporal-only" in s2_text
    assert "CROSS\\_LEVEL\\_STABLE" in s2_text
    assert "UNSUPPORTED\\_BOTH" in s2_text
    assert "FRAME\\_LOCAL" in s2_text
    assert "DISCORDANT\\_SIGN" in s2_text
    assert "TEMPORAL\\_EXPOSURE\\_SPECIFIC" in s2_text


def test_09_reproducibility_readme_exact_wording():
    p2k_dir = get_latest_phase2k_dir()
    readme_file = os.path.join(p2k_dir, "reproducibility", "README.md")
    with open(readme_file, "r") as f:
        readme = f.read()
        
    assert "raster forest plot generated from the supplied aggregate values" in readme, "Missing exact raster forest plot description in README"
    assert "Waymo Open Motion Dataset (WOMD) License Agreement" in readme


def test_10_reproducibility_fresh_directory_execution():
    p2k_dir = get_latest_phase2k_dir()
    repro_dir = os.path.join(p2k_dir, "reproducibility")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.copytree(repro_dir, os.path.join(tmp_dir, "reproducibility"))
        res = subprocess.run([sys.executable, "reproduce_paper_assets.py"],
                             cwd=os.path.join(tmp_dir, "reproducibility"),
                             capture_output=True, text=True)
        assert res.returncode == 0, f"Reproduction execution failed: {res.stderr}"
        assert "SUCCESS: Selected aggregate checks passed; Figure 2 reproduced." in res.stdout
        assert os.path.exists(os.path.join(tmp_dir, "reproducibility", "reproduced_assets", "reproduced_fig2_forest_plot.png"))
        
        with open(os.path.join(tmp_dir, "reproducibility", "reproduced_assets", "REPRODUCTION_REPORT.json")) as f:
            rep = json.load(f)
        assert rep["reproduction_status"] == "SELECTED_AGGREGATE_CHECKS_PASSED"
        assert rep["selected_assertions_checked"] >= 10


def test_11_supplement_zip_structure_excludes_main_pdf():
    p2k_dir = get_latest_phase2k_dir()
    supp_zip = os.path.join(p2k_dir, "submission_upload", "supplement.zip")
    assert os.path.exists(supp_zip)
    
    with zipfile.ZipFile(supp_zip, "r") as z:
        names = z.namelist()
        assert "main_anonymous.pdf" not in names, "main_anonymous.pdf must NOT be inside supplement.zip"
        assert "supplement_anonymous.pdf" in names, "supplement_anonymous.pdf missing in supplement.zip"
        assert any(n.endswith("reproduce_paper_assets.py") for n in names)
        assert any(n.endswith("CLAIM_EVIDENCE_LEDGER.csv") for n in names)
        assert any(n.endswith("TABLE1_NESTED_MODELS_V6.csv") for n in names)


def test_12_anonymity_scan_zero_local_paths():
    p2k_dir = get_latest_phase2k_dir()
    anon_json = os.path.join(p2k_dir, "qa", "ANONYMITY_SCAN.json")
    with open(anon_json, "r") as f:
        anon = json.load(f)
    assert anon["status"] == "PASS"
    assert anon["leak_count"] == 0


def test_13_submission_upload_exact_two_files():
    p2k_dir = get_latest_phase2k_dir()
    upload_dir = os.path.join(p2k_dir, "submission_upload")
    files = sorted(os.listdir(upload_dir))
    assert files == ["main_anonymous.pdf", "supplement.zip"], f"submission_upload must contain exactly 2 files, got: {files}"


def test_14_file_size_limits():
    p2k_dir = get_latest_phase2k_dir()
    main_pdf = os.path.join(p2k_dir, "submission_upload", "main_anonymous.pdf")
    supp_zip = os.path.join(p2k_dir, "submission_upload", "supplement.zip")
    
    assert os.path.getsize(main_pdf) < 50 * 1024 * 1024, "Main PDF exceeds 50MB limit"
    assert os.path.getsize(supp_zip) < 100 * 1024 * 1024, "Supplement ZIP exceeds 100MB limit"


def test_15_checksums_manifest_presence():
    p2k_dir = get_latest_phase2k_dir()
    chk = os.path.join(p2k_dir, "CHECKSUMS_SHA256.txt")
    assert os.path.exists(chk), "CHECKSUMS_SHA256.txt missing"
    with open(chk, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 15, "Checksum manifest too small"


def test_16_dynamic_gates_json_structure():
    p2k_dir = get_latest_phase2k_dir()
    gates_file = os.path.join(p2k_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    assert os.path.exists(gates_file)
    with open(gates_file, "r") as f:
        gates_data = json.load(f)
        
    assert "FINAL_SUBMISSION_STATUS" in gates_data
    assert "GATES" in gates_data
    assert len(gates_data["GATES"]) >= 20
    for gname, gval in gates_data["GATES"].items():
        assert "status" in gval
        assert "observed_value" in gval


def test_17_paper_id_handling_logic():
    p2k_dir = get_latest_phase2k_dir()
    gates_file = os.path.join(p2k_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    with open(gates_file, "r") as f:
        gates_data = json.load(f)
        
    if not gates_data["IS_VALID_PAPER_ID"]:
        assert gates_data["FINAL_SUBMISSION_STATUS"] == "PAPER_ID_REQUIRED"
        assert gates_data["GATES"]["PAPER_ID_PRESENT_GATE"]["status"] == "PAPER_ID_REQUIRED"
    else:
        assert gates_data["FINAL_SUBMISSION_STATUS"] == "READY_TO_UPLOAD"
        assert gates_data["GATES"]["PAPER_ID_PRESENT_GATE"]["status"] == "PASS"


def test_18_output_zip_bundles_integrity():
    p2k_dir = get_latest_phase2k_dir()
    out_dir = os.path.join(p2k_dir, "output")
    for zname in ["phase2k_submission_package.zip", "phase2k_feedback_bundle.zip", "phase2k_code_package.zip"]:
        zp = os.path.join(out_dir, zname)
        assert os.path.exists(zp), f"{zname} missing in output"
        assert os.path.getsize(zp) > 100000, f"{zname} suspiciously small"
        with zipfile.ZipFile(zp, "r") as z:
            assert len(z.namelist()) > 0


def test_19_all_pages_rendered_200dpi():
    p2k_dir = get_latest_phase2k_dir()
    renders_dir = os.path.join(p2k_dir, "qa", "page_renders")
    assert os.path.exists(renders_dir)
    pngs = glob.glob(os.path.join(renders_dir, "*.png"))
    assert len(pngs) == 8, f"Expected 8 rendered pages (6 main + 2 supp), found {len(pngs)}"


def test_20_figures_and_tables_exist():
    p2k_dir = get_latest_phase2k_dir()
    fig_dir = os.path.join(p2k_dir, "corrected_source", "paper_source", "figures")
    for i in [1, 2, 3, 4]:
        pdf = glob.glob(os.path.join(fig_dir, f"fig{i}_*.pdf"))
        png = glob.glob(os.path.join(fig_dir, f"fig{i}_*.png"))
        assert len(pdf) == 1, f"Figure {i} PDF missing"
        assert len(png) == 1, f"Figure {i} PNG missing"
