# Master Runbook: Complete End-to-End Reproduction Guide

This runbook documents how to reproduce the entire scientific workflow from scratch, even if all other workspace files were removed.

---

## 1. Environment Setup
```bash
# Option A: Conda environment (recommended)
conda create -n r2env python=3.10 -y
conda activate r2env
pip install -r 01_environment_and_setup/phase2-core.txt
pip install -r 01_environment_and_setup/phase2-model.txt

# Option B: Docker container
docker compose -f 01_environment_and_setup/compose.yaml up -d
```

---

## 2. Step-by-Step Pipeline Execution

### Step 1: Raw WOMD Parsing and Feature Extraction
```bash
python 02_core_geometry_and_womd/parser.py --womd_dir /path/to/raw/womd/ --out_dir work/parsed/
python 03_feature_engineering_odd/odd_feature_engine.py --input_dir work/parsed/ --out_dir work/features/
```

### Step 2: Cohort Splitting
```bash
python 04_cohort_split_manifests/split_manifest.py --input_dir work/features/ --manifest 04_cohort_split_manifests/split_manifest.json
python 04_cohort_split_manifests/split_near_duplicate_audit.py --features_dir work/features/
```

### Step 3: Model Training
```bash
python 05_model_training_and_frozen/train_phase2_models.py --dev_features work/features/dev/ --save_dir 05_model_training_and_frozen/models_frozen/
```

### Step 4: One-Shot Sealed Holdout Evaluation & Bootstrap
```bash
python 06_holdout_eval_and_bootstrap/open_holdout_once.py --holdout_features work/features/holdout/ --models_dir 05_model_training_and_frozen/models_frozen/
python 06_holdout_eval_and_bootstrap/r2_bootstrap.py --predictions work/holdout_predictions.parquet --iterations 1000
```

### Step 5: Scenario Profiles, Tau Sensitivity, and KPI Alignment
```bash
python 07_scenario_profiles_and_kpi/temporal_robustness.py --input_dir work/features/
python 07_scenario_profiles_and_kpi/scenario_diagnostics.py --input_dir work/features/
python 07_scenario_profiles_and_kpi/construct_validity.py --input_dir work/features/
```

### Step 6: Master Packaging, Paper Compilation & Visual QA
```bash
# Provide real Paper ID or leave empty for default placeholder verification:
WACV_PAPER_ID=1234 python 11_master_pipeline_and_engine/phase2l_master_engine.py
```

### Step 7: Automated Asset Verification
```bash
python 11_master_pipeline_and_engine/reproduce_paper_assets.py
pytest -v 13_test_suite_and_verification/
```

---

## 3. Authoritative Locked Results Summary
- $M_P$ (Physical Baseline): $\text{AP} = 0.3224$
- $M_{P+E_{\text{all}}}$ (Primary Full): $\text{AP} = 0.3370$
- Incremental Contrast: $\Delta\text{AP} = +0.0147$ ($95\%$ CI $[+0.0005, +0.0285]$)
- Feature Confirmation: $10/13$ confirmed, $13/13$ sign concordant
- Supportive KPI Alignment: $\rho = +0.1347$, Cohen's $d = +0.3451$
