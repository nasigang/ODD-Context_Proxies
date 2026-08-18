#!/usr/bin/env python3
"""
Unit Test Suite for WACV 2027 Phase 2L — Bibliographic Integrity and Real Paper-ID Seal
=======================================================================================
Validates that:
  1. All 16 cited references are verified against primary sources with BIBLIOGRAPHY_PROVENANCE.csv.
  2. Task B text replacements in Introduction and Related Work are strictly applied.
  3. Scanlon is cleanly removed from rendered citations when no longer cited.
  4. LaTeX compilation yields 0 undefined citations, 0 undefined references, 0 overfull hboxes.
  5. Exact page counts (6 main, 2 supplement) and 0 Type 3 fonts.
  6. Reproducibility package executes with returncode 0 in an isolated directory.
  7. supplement.zip strictly excludes main_anonymous.pdf.
  8. Anonymity scan detects 0 local paths or names.
  9. submission_upload/ strictly contains exactly 2 files.
 10. Paper ID gate behaves correctly (PAPER_ID_REQUIRED on placeholder, READY_TO_UPLOAD on real ID).
"""

import os
import re
import sys
import glob
import json
import zipfile
import subprocess
import tempfile
import pandas as pd
import pytest


WORKSPACE = "/home/kiapi/waymo_motion_project"
PYTHON_BIN = "/home/kiapi/miniconda3/envs/r2env/bin/python"


@pytest.fixture(scope="session")
def latest_phase2l_work_dir():
    # Find latest phase2l directory or run engine if none exists
    p2l_dirs = sorted(glob.glob(os.path.join(WORKSPACE, "work", "phase2l_bibliographic_seal_*")))
    if not p2l_dirs:
        res = subprocess.run([PYTHON_BIN, os.path.join(WORKSPACE, "phase2_womd", "phase2l_master_engine.py")],
                             cwd=WORKSPACE, capture_output=True, text=True)
        assert res.returncode == 0, f"Engine execution failed: {res.stderr}"
        p2l_dirs = sorted(glob.glob(os.path.join(WORKSPACE, "work", "phase2l_bibliographic_seal_*")))
    return p2l_dirs[-1]


def test_01_input_inventory_presence_and_hashes(latest_phase2l_work_dir):
    inv_csv = os.path.join(latest_phase2l_work_dir, "qa", "INPUT_INVENTORY_SHA256.csv")
    assert os.path.exists(inv_csv), "INPUT_INVENTORY_SHA256.csv missing"
    df = pd.read_csv(inv_csv)
    assert len(df) >= 3, "Expected at least 3 inventoried packages"
    assert "sha256" in df.columns and "filename" in df.columns
    for sha in df["sha256"]:
        assert len(sha) == 64, f"Invalid SHA-256 hash length: {sha}"


def test_02_bibliographic_provenance_and_verification(latest_phase2l_work_dir):
    prov_csv = os.path.join(latest_phase2l_work_dir, "qa", "BIBLIOGRAPHY_PROVENANCE.csv")
    assert os.path.exists(prov_csv), "BIBLIOGRAPHY_PROVENANCE.csv missing"
    df = pd.read_csv(prov_csv)
    
    expected_cols = ["citation_key", "title", "authors", "venue", "year", "doi_or_url", "primary_source_url", "verified"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c} in BIBLIOGRAPHY_PROVENANCE.csv"
        
    assert (df["verified"] == True).all(), "Not all bibliography entries are verified == True"
    assert len(df) == 16, f"Expected 16 verified entries, got {len(df)}"
    
    # Check key explicitly mandated entries
    mandated_keys = ["caesar2020nuscenes", "ettinger2021large", "westhofen2023criticality",
                     "stoler2024safeshift", "puphal2025risk", "weng2023joint"]
    for k in mandated_keys:
        assert k in df["citation_key"].values, f"Mandated key {k} missing from provenance"


def test_03_task_b_citation_text_replacements(latest_phase2l_work_dir):
    intro_tex = os.path.join(latest_phase2l_work_dir, "corrected_source", "paper_source", "sec", "1_intro.tex")
    with open(intro_tex, "r") as f:
        intro_txt = f.read()
        
    expected_intro_start = "Pairwise TTC surrogates characterize instantaneous criticality between two actors, while recorded driving scenes contain additional nearby actors and contextual conditions"
    assert expected_intro_start in intro_txt, "Task B Introduction replacement sentence missing"
    assert "scanlon2021waymo" not in intro_txt, "scanlon2021waymo must not be in intro"

    rw_tex = os.path.join(latest_phase2l_work_dir, "corrected_source", "paper_source", "sec", "2_related_work.tex")
    with open(rw_tex, "r") as f:
        rw_txt = f.read()
        
    expected_rw_end = "Together, these studies motivate interaction-aware evaluation; our focus is complementary: we audit whether dataset-observable current-frame ODD-context proxies add predictive information beyond a focal SDC--actor kinematic baseline for an ego-centric TTC target."
    assert expected_rw_end in rw_txt, "Task B Related Work replacement sentence missing"


def test_04_scanlon_removed_when_uncited(latest_phase2l_work_dir):
    # Check that scanlon is not cited in any section
    sec_dir = os.path.join(latest_phase2l_work_dir, "corrected_source", "paper_source", "sec")
    for tf in glob.glob(os.path.join(sec_dir, "*.tex")):
        with open(tf, "r") as f:
            content = f.read()
        assert "scanlon" not in content, f"Found scanlon cited in {tf}"


def test_05_compilation_zero_undefined_citations_and_references(latest_phase2l_work_dir):
    qa_json = os.path.join(latest_phase2l_work_dir, "qa", "LATEX_COMPILE_QA.json")
    assert os.path.exists(qa_json), "LATEX_COMPILE_QA.json missing"
    with open(qa_json, "r") as f:
        qa = json.load(f)
    assert qa["main_pdf"]["undefined_citations"] == 0, f"Found undefined citations: {qa['main_pdf']['undefined_citations']}"
    assert qa["main_pdf"]["undefined_references"] == 0, f"Found undefined references: {qa['main_pdf']['undefined_references']}"


def test_06_table1_tabularx_zero_overfull_hboxes(latest_phase2l_work_dir):
    qa_json = os.path.join(latest_phase2l_work_dir, "qa", "LATEX_COMPILE_QA.json")
    with open(qa_json, "r") as f:
        qa = json.load(f)
    assert qa["main_pdf"]["overfull_hboxes"] == 0, f"Main PDF has {qa['main_pdf']['overfull_hboxes']} overfull hboxes"
    assert qa["supplement_pdf"]["overfull_hboxes"] == 0, f"Supplement PDF has {qa['supplement_pdf']['overfull_hboxes']} overfull hboxes"


def test_07_page_limits_exact(latest_phase2l_work_dir):
    qa_json = os.path.join(latest_phase2l_work_dir, "qa", "LATEX_COMPILE_QA.json")
    with open(qa_json, "r") as f:
        qa = json.load(f)
    assert qa["main_pdf"]["pages"] <= 8, f"Main PDF exceeds 8 pages: {qa['main_pdf']['pages']}"
    assert qa["main_pdf"]["pages"] == 6, f"Main PDF expected exactly 6 pages, got {qa['main_pdf']['pages']}"
    assert qa["supplement_pdf"]["pages"] == 2, f"Supplement PDF expected exactly 2 pages, got {qa['supplement_pdf']['pages']}"


def test_08_font_properties_zero_type3(latest_phase2l_work_dir):
    qa_json = os.path.join(latest_phase2l_work_dir, "qa", "LATEX_COMPILE_QA.json")
    with open(qa_json, "r") as f:
        qa = json.load(f)
    assert qa["main_pdf"]["has_type3_fonts"] is False, "Type 3 fonts detected in main PDF"
    assert qa["supplement_pdf"]["has_type3_fonts"] is False, "Type 3 fonts detected in supplement PDF"


def test_09_reproducibility_readme_exact_wording(latest_phase2l_work_dir):
    readme_file = os.path.join(latest_phase2l_work_dir, "reproducibility", "README.md")
    assert os.path.exists(readme_file)
    with open(readme_file, "r") as f:
        content = f.read()
    assert "raster forest plot generated from the supplied aggregate values" in content
    assert "Waymo Open Motion Dataset (WOMD) License Agreement" in content


def test_10_reproducibility_fresh_directory_execution(latest_phase2l_work_dir):
    repro_dir = os.path.join(latest_phase2l_work_dir, "reproducibility")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_dir = os.path.join(tmp_dir, "repro")
        import shutil
        shutil.copytree(repro_dir, dest_dir)
        res = subprocess.run([PYTHON_BIN, "reproduce_paper_assets.py"],
                             cwd=dest_dir, capture_output=True, text=True)
        assert res.returncode == 0, f"Execution failed: {res.stderr}"
        assert "SUCCESS: Selected aggregate checks passed" in res.stdout
        assert os.path.exists(os.path.join(dest_dir, "reproduced_assets", "reproduced_fig2_forest_plot.png"))
        assert os.path.exists(os.path.join(dest_dir, "reproduced_assets", "REPRODUCTION_REPORT.json"))


def test_11_supplement_zip_structure_excludes_main_pdf(latest_phase2l_work_dir):
    supp_zip = os.path.join(latest_phase2l_work_dir, "submission_upload", "supplement.zip")
    assert os.path.exists(supp_zip), "supplement.zip missing"
    with zipfile.ZipFile(supp_zip, "r") as z:
        names = z.namelist()
        assert "main_anonymous.pdf" not in names, "main_anonymous.pdf must NOT be in supplement.zip"
        assert "supplement_anonymous.pdf" in names, "supplement_anonymous.pdf missing in supplement.zip"
        assert any(n.endswith("reproduce_paper_assets.py") for n in names)
        assert any(n.endswith("TABLE1_NESTED_MODELS_V6.csv") for n in names)


def test_12_anonymity_scan_zero_local_paths(latest_phase2l_work_dir):
    anon_json = os.path.join(latest_phase2l_work_dir, "qa", "ANONYMITY_SCAN.json")
    assert os.path.exists(anon_json)
    with open(anon_json, "r") as f:
        report = json.load(f)
    assert report["status"] == "PASS", f"Anonymity scan failed: {report['leaks']}"
    assert report["leak_count"] == 0


def test_13_submission_upload_exact_two_files(latest_phase2l_work_dir):
    upload_dir = os.path.join(latest_phase2l_work_dir, "submission_upload")
    files = sorted(os.listdir(upload_dir))
    assert files == ["main_anonymous.pdf", "supplement.zip"], f"Expected exactly 2 files, found: {files}"


def test_14_file_size_limits(latest_phase2l_work_dir):
    main_pdf = os.path.join(latest_phase2l_work_dir, "submission_upload", "main_anonymous.pdf")
    supp_zip = os.path.join(latest_phase2l_work_dir, "submission_upload", "supplement.zip")
    assert os.path.getsize(main_pdf) < 20 * 1024 * 1024, "main_anonymous.pdf exceeds 20MB"
    assert os.path.getsize(supp_zip) < 50 * 1024 * 1024, "supplement.zip exceeds 50MB"


def test_15_checksums_manifest_presence(latest_phase2l_work_dir):
    cs_file = os.path.join(latest_phase2l_work_dir, "CHECKSUMS_SHA256.txt")
    assert os.path.exists(cs_file)
    with open(cs_file, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 15, "Checksum manifest too small"


def test_16_dynamic_gates_json_structure(latest_phase2l_work_dir):
    gates_json = os.path.join(latest_phase2l_work_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    assert os.path.exists(gates_json)
    with open(gates_json, "r") as f:
        gates_data = json.load(f)
    assert "FINAL_SUBMISSION_STATUS" in gates_data
    assert "GATES" in gates_data
    assert len(gates_data["GATES"]) >= 20


def test_17_paper_id_handling_logic():
    # Test execution with a real mock Paper ID
    env = os.environ.copy()
    env["WACV_PAPER_ID"] = "9876"
    res = subprocess.run([PYTHON_BIN, os.path.join(WORKSPACE, "phase2_womd", "phase2l_master_engine.py")],
                         env=env, cwd=WORKSPACE, capture_output=True, text=True)
    assert res.returncode == 0, f"Engine failed with valid ID: {res.stderr}"
    
    # Locate latest created workdir
    p2l_dirs = sorted(glob.glob(os.path.join(WORKSPACE, "work", "phase2l_bibliographic_seal_*")))
    valid_work_dir = p2l_dirs[-1]
    
    gates_file = os.path.join(valid_work_dir, "qa", "FINAL_SUBMISSION_GATES.json")
    with open(gates_file, "r") as f:
        gdata = json.load(f)
    assert gdata["FINAL_SUBMISSION_STATUS"] == "READY_TO_UPLOAD"
    assert gdata["WACV_PAPER_ID"] == "9876"
    
    # Verify zero '*****' remains
    res_txt = subprocess.run(["pdftotext", os.path.join(valid_work_dir, "submission_upload", "main_anonymous.pdf"), "-"],
                             capture_output=True, text=True, check=True)
    assert "*****" not in res_txt.stdout, "Placeholder '*****' found in main_anonymous.pdf"


def test_18_output_zip_bundles_integrity(latest_phase2l_work_dir):
    out_dir = os.path.join(latest_phase2l_work_dir, "output")
    for zb in ["phase2l_submission_package.zip", "phase2l_feedback_bundle.zip", "phase2l_code_package.zip"]:
        zp = os.path.join(out_dir, zb)
        assert os.path.exists(zp), f"{zb} missing in output"
        with zipfile.ZipFile(zp, "r") as z:
            assert len(z.namelist()) > 0


def test_19_all_pages_rendered_200dpi(latest_phase2l_work_dir):
    renders_dir = os.path.join(latest_phase2l_work_dir, "qa", "page_renders")
    pngs = sorted(glob.glob(os.path.join(renders_dir, "*.png")))
    assert len(pngs) == 8, f"Expected 8 rendered PNG pages (6 main + 2 supp), found {len(pngs)}"


def test_20_figures_and_tables_exist(latest_phase2l_work_dir):
    vqa_csv = os.path.join(latest_phase2l_work_dir, "qa", "PAGE_VISUAL_QA.csv")
    assert os.path.exists(vqa_csv)
    df = pd.read_csv(vqa_csv)
    assert len(df) >= 10
    assert (df["status"] == "VERIFIED_PERFECT").all()
