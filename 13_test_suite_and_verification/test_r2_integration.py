#!/usr/bin/env python3
"""
R2 Production Integration Tests — tests production entrypoints, NOT just math units.

Per §7 of production integration correction:
- Import/CLI smoke
- End-to-end synthetic production path
- Regression tests (old API, empty freeze, etc.)
- R1 target tests (overlap, circle, validation path)
"""
import hashlib
import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ═══════════════════════════════════════════════════════════════
# §7.1 Import/CLI Smoke
# ═══════════════════════════════════════════════════════════════

class TestImportSmoke:
    """All production modules import cleanly."""

    def test_import_r2_censored_likelihood(self):
        from phase2_womd.r2_censored_likelihood import (
            censored_lognormal_nll, nll_and_grad, predict_exceedance_prob,
            censored_nll_components, NLL_LABELS, _stable_mills_ratio
        )

    def test_import_r2_models(self):
        from phase2_womd.r2_models import (
            CensoredLogNormalAFT, create_model_suite,
            Z_STATE_FEATURES, C_CONTEXT_FEATURES, M4_INTERACTIONS
        )

    def test_import_train_r2_models(self):
        from phase2_womd.train_r2_models import (
            train_production, _train_models_internal,
            GateToken, ModelIdentifiabilityError
        )

    def test_import_evaluate_frozen_r2(self):
        from phase2_womd.evaluate_frozen_r2 import (
            FrozenEvaluator, SplitAccessError,
            ALLOWED_SPLITS_STAGE_A, BLOCKED_SPLITS
        )

    def test_import_prepare_r2_table(self):
        from phase2_womd.prepare_r2_table import (
            prepare_r2_table, load_frame_targets,
            apply_risk_set_filter, apply_missing_policy
        )

    def test_import_r2_split(self):
        from phase2_womd.r2_split import (
            assign_split, generate_split_membership,
            deterministic_split_hash, save_frozen_membership
        )

    def test_import_r2_input_gate(self):
        from phase2_womd.r2_input_gate import R2InputGate

    def test_import_select_and_freeze(self):
        from phase2_womd.select_and_freeze_r2 import (
            freeze_model_config, verify_frozen, FreezeError
        )

    def test_import_r2_bootstrap(self):
        from phase2_womd.r2_bootstrap import (
            scenario_block_paired_bootstrap, compute_conditional_brier,
            DELTA_SIGN_CONVENTION
        )

    def test_import_open_holdout(self):
        from phase2_womd.open_holdout_once import (
            HoldoutGuard, HoldoutGuardError
        )

    def test_no_deleted_class_import(self):
        """CensoredGaussianModel must not exist in r2_models."""
        import phase2_womd.r2_models as m
        assert not hasattr(m, "CensoredGaussianModel"), \
            "Deleted CensoredGaussianModel still importable"

    def test_no_deleted_function_import(self):
        """censored_gaussian_nll_components must not exist."""
        import phase2_womd.r2_censored_likelihood as ll
        assert not hasattr(ll, "censored_gaussian_nll_components"), \
            "Deleted censored_gaussian_nll_components still importable"


# ═══════════════════════════════════════════════════════════════
# §7.2 End-to-End Synthetic Production Path
# ═══════════════════════════════════════════════════════════════

def _make_synthetic_r1_fixture(path, n=200, n_scenarios=10, seed=42):
    """Create a synthetic accepted-R1 OBB frame target fixture."""
    rng = np.random.RandomState(seed)
    scenarios = [f"sc_{i:04d}" for i in range(n_scenarios)]
    sids = rng.choice(scenarios, n)
    times = rng.randint(0, 91, n)

    statuses = rng.choice(
        ["future_contact_event", "right_censored", "current_geometry_overlap", "no_exposure"],
        n, p=[0.15, 0.55, 0.05, 0.25])
    ttc = rng.uniform(0.5, 9.5, n)
    censored = statuses == "right_censored"
    ttc[censored] = 10.0
    ttc[statuses == "current_geometry_overlap"] = 0.0
    ttc[statuses == "no_exposure"] = np.nan
    overlap = statuses == "current_geometry_overlap"
    right_cens = censored.astype(bool)

    t = pa.table({
        "scenario_id": sids,
        "time_index": times,
        "ego_track_id": np.ones(n, dtype=int),
        "ttc_obb_swept_s": ttc.astype(np.float64),
        "target_status": statuses,
        "right_censored": right_cens,
        "overlap_now_flag": overlap.astype(bool),
        "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"] * n,
        "ego_speed_mps": rng.uniform(0, 30, n).astype(np.float64),
        "ego_accel_mps2": rng.uniform(-3, 3, n).astype(np.float64),
        "ego_yaw_rate_rps": rng.uniform(-0.5, 0.5, n).astype(np.float64),
        "n_eligible_pairs": rng.randint(0, 10, n).astype(np.float64),
        "min_pair_distance_m": rng.uniform(5, 60, n).astype(np.float64),
        "max_closing_speed_mps": rng.uniform(0, 15, n).astype(np.float64),
        "traffic_n_valid_agents": rng.randint(1, 30, n).astype(np.float64),
    })
    pq.write_table(t, path)
    return scenarios


class TestEndToEndSynthetic:
    """Full production pipeline on synthetic fixture."""

    def test_full_path(self, tmp_path):
        from phase2_womd.prepare_r2_table import prepare_r2_table
        from phase2_womd.train_r2_models import (
            _train_models_internal, GateToken
        )
        from phase2_womd.evaluate_frozen_r2 import FrozenEvaluator
        from phase2_womd.r2_models import CensoredLogNormalAFT
        from phase2_womd.select_and_freeze_r2 import freeze_model_config, verify_frozen
        from phase2_womd.r2_bootstrap import DELTA_SIGN_CONVENTION

        # 1. Create fixture
        frame_path = str(tmp_path / "r1_frames.parquet")
        scenarios = _make_synthetic_r1_fixture(frame_path)

        # 2. Prepare table (generates membership from fixture)
        output_dir = str(tmp_path / "output")
        df_primary, membership, preproc, feat_names = prepare_r2_table(
            frame_path, output_dir)

        assert len(df_primary) > 0
        assert "split" in df_primary.columns
        assert (df_primary["overlap_now_flag"] == False).all()

        # 3. Train (with gate token)
        df_train = df_primary[df_primary["split"] == "train"]
        gate_token = GateToken("synthetic_test", "synthetic_hash")
        models, report = _train_models_internal(
            df_train, feat_names, preproc, output_dir, gate_token)

        # Verify at least M0, M1, M3 fitted
        for m in ["M0", "M1", "M3"]:
            assert report["models"][m]["status"] == "FIT_SUCCESS"

        # 4. Artifact save/load parity
        m1_path = os.path.join(output_dir, "model_artifacts", "model_M1.pkl")
        m1_loaded = CensoredLogNormalAFT.load(m1_path)
        X_train_vals = preproc["scaler"].transform(df_train[feat_names].values)
        mu_orig, _ = models["M1"].predict(
            X_train_vals[:, [feat_names.index(f) for f in models["M1"].feature_names]])
        mu_loaded, _ = m1_loaded.predict(
            X_train_vals[:, [feat_names.index(f) for f in m1_loaded.feature_names]])
        np.testing.assert_array_almost_equal(mu_orig, mu_loaded)

        # 5. Train-only frozen evaluation
        evaluator = FrozenEvaluator(
            os.path.join(output_dir, "model_artifacts"),
            os.path.join(output_dir, "model_artifacts", "preprocessing_scaler.pkl"),
            stage="A")
        for m in ["M0", "M1", "M3"]:
            evaluator.load_model(m)
        evaluator.load_preprocessing()

        y_ttc = df_train["ttc_obb_swept_s"].values
        censored = df_train["right_censored"].values.astype(bool)
        sids = df_train["scenario_id"].values
        eval_report = evaluator.evaluate_on_split(
            X_train_vals, y_ttc, censored, sids, "train")
        assert eval_report["FIT_CALLED_DURING_EVAL"] is False

        # 6. Primary comparison (M1 vs M3, delta = NLL_M1 - NLL_M3)
        comparison = evaluator.run_primary_comparison(
            X_train_vals, y_ttc, censored, sids, n_boot=50)
        assert "NLL_M1 - NLL_M3" in comparison["sign_convention"]

        # 7. Synthetic freeze/verify
        frozen = freeze_model_config(
            os.path.join(output_dir, "model_artifacts"),
            os.path.join(output_dir, "model_artifacts", "preprocessing_scaler.pkl"),
            feat_names, output_dir)
        assert frozen["status"] == "FROZEN"
        assert "Jacobian" in frozen["nll_label"]

        ok, issues = verify_frozen(
            os.path.join(output_dir, "frozen_config.json"),
            os.path.join(output_dir, "model_artifacts"),
            os.path.join(output_dir, "model_artifacts", "preprocessing_scaler.pkl"))
        assert ok, f"Freeze verify failed: {issues}"


# ═══════════════════════════════════════════════════════════════
# §7.3 Regression Tests
# ═══════════════════════════════════════════════════════════════

class TestRegressionOldAPI:
    """Old API calls must fail or not exist."""

    def test_no_CensoredGaussianModel(self):
        import phase2_womd.r2_models as m
        assert not hasattr(m, "CensoredGaussianModel")

    def test_no_censored_gaussian_nll_components(self):
        import phase2_womd.r2_censored_likelihood as ll
        assert not hasattr(ll, "censored_gaussian_nll_components")

    def test_trainer_gate_bypass(self):
        """Direct call to _train_models_internal without GateToken fails."""
        from phase2_womd.train_r2_models import _train_models_internal
        with pytest.raises(TypeError, match="gate_token"):
            _train_models_internal(None, None, None, None, gate_token="not_a_token")


class TestRegressionFreeze:
    """Freeze fail-closed tests."""

    def test_empty_model_dir_fails(self, tmp_path):
        from phase2_womd.select_and_freeze_r2 import freeze_model_config, FreezeError
        mdir = str(tmp_path / "empty_models")
        os.makedirs(mdir)
        with pytest.raises(FreezeError, match="No model"):
            freeze_model_config(mdir, str(tmp_path / "x.pkl"), ["f1"], str(tmp_path))

    def test_missing_preprocessing_fails(self, tmp_path):
        from phase2_womd.select_and_freeze_r2 import freeze_model_config, FreezeError
        from phase2_womd.r2_models import CensoredLogNormalAFT
        mdir = str(tmp_path / "models"); os.makedirs(mdir)
        for name in ["M0", "M1", "M3"]:
            feats = ["f1"] if name != "M0" else []
            m = CensoredLogNormalAFT(name, feats)
            X = np.zeros((50, 0)) if name == "M0" else np.random.randn(50, 1)
            m.fit(X, np.exp(np.random.randn(50)), np.random.rand(50) > 0.7)
            m.save(os.path.join(mdir, f"model_{name}.pkl"))
        with pytest.raises(FreezeError, match="not found"):
            freeze_model_config(mdir, "/nonexistent.pkl", ["f1"], str(tmp_path))

    def test_wrong_nll_label_detected(self, tmp_path):
        from phase2_womd.select_and_freeze_r2 import verify_frozen
        frozen = {
            "nll_label": "Gaussian NLL on log(TTC) (no Jacobian)",
            "models": {}, "preprocessing_hash": ""
        }
        fp = str(tmp_path / "frozen.json")
        with open(fp, "w") as f:
            json.dump(frozen, f)
        ok, issues = verify_frozen(fp, str(tmp_path), str(tmp_path / "x.pkl"))
        assert not ok
        assert any("NLL label" in i for i in issues)


class TestRegressionEvaluator:
    """Evaluator restriction tests."""

    def test_holdout_blocked(self, tmp_path):
        from phase2_womd.evaluate_frozen_r2 import FrozenEvaluator, SplitAccessError
        ev = FrozenEvaluator(str(tmp_path), str(tmp_path / "x.pkl"))
        with pytest.raises(SplitAccessError, match="BLOCKED"):
            ev._check_split("internal_holdout")

    def test_external_blocked(self, tmp_path):
        from phase2_womd.evaluate_frozen_r2 import FrozenEvaluator, SplitAccessError
        ev = FrozenEvaluator(str(tmp_path), str(tmp_path / "x.pkl"))
        with pytest.raises(SplitAccessError, match="BLOCKED"):
            ev._check_split("external_test")

    def test_internal_val_blocked_stage_a(self, tmp_path):
        from phase2_womd.evaluate_frozen_r2 import FrozenEvaluator, SplitAccessError
        ev = FrozenEvaluator(str(tmp_path), str(tmp_path / "x.pkl"), stage="A")
        with pytest.raises(SplitAccessError, match="not allowed"):
            ev._check_split("internal_val")

    def test_brier_fail_if_censor_lt_tau(self, tmp_path):
        """Brier score must be None when censor_time < tau."""
        from phase2_womd.r2_bootstrap import compute_conditional_brier
        from phase2_womd.r2_models import CensoredLogNormalAFT
        m = CensoredLogNormalAFT("test", ["f1"])
        m.fit(np.random.randn(50, 1), np.exp(np.random.randn(50)),
              np.random.rand(50) > 0.7)
        X = np.random.randn(20, 1)
        y = np.exp(np.random.randn(20))
        c = np.random.rand(20) > 0.7
        ct = np.full(20, 2.0)  # censor at 2s
        result = compute_conditional_brier(m, X, y, c, tau=5.0, censor_time=ct)
        # censor_time (2.0) < tau (5.0) → brier not valid
        assert result["censor_time_ge_tau"] is False

    def test_delta_sign_locked(self):
        from phase2_womd.r2_bootstrap import DELTA_SIGN_CONVENTION
        assert "NLL_M1 - NLL_M3" in DELTA_SIGN_CONVENTION
        assert "positive = M3 better" in DELTA_SIGN_CONVENTION


class TestRegressionOptimizer:

    def test_all_censored_identifiability_error(self):
        from phase2_womd.train_r2_models import (
            _train_models_internal, GateToken, ModelIdentifiabilityError
        )
        df = pd.DataFrame({
            "ego_speed_mps": [1.0]*10, "ego_accel_mps2": [0.0]*10,
            "ego_yaw_rate_rps": [0.0]*10, "n_eligible_pairs": [1.0]*10,
            "min_pair_distance_m": [20.0]*10, "max_closing_speed_mps": [5.0]*10,
            "traffic_n_valid_agents": [10.0]*10,
            "ttc_obb_swept_s": [10.0]*10,
            "right_censored": [True]*10,
        })
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feat = ["ego_speed_mps", "ego_accel_mps2", "ego_yaw_rate_rps",
                "n_eligible_pairs", "min_pair_distance_m", "max_closing_speed_mps",
                "traffic_n_valid_agents"]
        scaler.fit(df[feat].values)
        preproc = {"scaler": scaler}
        token = GateToken("test", "test")
        with pytest.raises(ModelIdentifiabilityError):
            _train_models_internal(df, feat, preproc, "/tmp/test_opt", token)

    def test_duplicated_split_implementation_removed(self):
        """prepare_r2_table must import split from r2_split, not define locally."""
        import inspect
        from phase2_womd import prepare_r2_table as mod
        # The function should be imported from r2_split, not defined in prepare_r2_table
        gsm = getattr(mod, "generate_split_membership", None)
        if gsm is not None:
            defining_module = gsm.__module__
            assert defining_module == "phase2_womd.r2_split", \
                f"generate_split_membership defined in {defining_module}, not r2_split"
        # Also verify no rng.shuffle in function bodies defined in this module
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if func.__module__ == "phase2_womd.prepare_r2_table":
                src = inspect.getsource(func)
                assert "rng.shuffle" not in src, \
                    f"rng.shuffle found in {name}() — use canonical r2_split.py"


# ═══════════════════════════════════════════════════════════════
# §7.4 R1 Target Tests
# ═══════════════════════════════════════════════════════════════

class TestR1Targets:

    def test_overlap_precedence(self, tmp_path):
        """Frame with ANY-pair overlap excluded from primary."""
        from phase2_womd.prepare_r2_table import apply_risk_set_filter
        df = pd.DataFrame({
            "scenario_id": ["a", "a"],
            "time_index": [0, 1],
            "target_status": ["future_contact_event", "current_geometry_overlap"],
            "overlap_now_flag": [False, True],
            "right_censored": [False, False],
        })
        primary, _, _ = apply_risk_set_filter(df)
        assert len(primary) == 1
        assert (primary["overlap_now_flag"] == False).all()

    def test_overlap_in_primary_row_fails(self, tmp_path):
        """Event row with overlap_now=True raises ValueError."""
        from phase2_womd.prepare_r2_table import apply_risk_set_filter
        df = pd.DataFrame({
            "scenario_id": ["a"],
            "time_index": [0],
            "target_status": ["future_contact_event"],
            "overlap_now_flag": [True],
            "right_censored": [False],
        })
        with pytest.raises(ValueError, match="overlap_now_flag"):
            apply_risk_set_filter(df)

    def test_circle_column_rejected(self, tmp_path):
        """Primary table rejects Circle/legacy columns."""
        from phase2_womd.prepare_r2_table import load_frame_targets
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "ttc_obb_swept_s": [3.0], "target_status": ["right_censored"],
            "right_censored": [True], "overlap_now_flag": [False],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"],
            "circle_ttc_s": [4.0],  # BANNED
        })
        fpath = str(tmp_path / "banned.parquet")
        pq.write_table(t, fpath)
        with pytest.raises(ValueError, match="banned"):
            load_frame_targets(fpath)

    def test_ttc_circle_s_also_rejected(self, tmp_path):
        """Alternate name ttc_circle_s also rejected."""
        from phase2_womd.prepare_r2_table import load_frame_targets
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "ttc_obb_swept_s": [3.0], "target_status": ["right_censored"],
            "right_censored": [True], "overlap_now_flag": [False],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"],
            "ttc_circle_s": [4.0],  # BANNED alternate name
        })
        fpath = str(tmp_path / "banned2.parquet")
        pq.write_table(t, fpath)
        with pytest.raises(ValueError, match="banned"):
            load_frame_targets(fpath)

    def test_missing_feature_fails(self, tmp_path):
        """Missing required feature column → FeatureIntegrityError."""
        from phase2_womd.prepare_r2_table import apply_missing_policy
        from phase2_womd.r2_feature_engineering import FeatureIntegrityError
        df = pd.DataFrame({"ego_speed_mps": [1.0]})
        with pytest.raises(FeatureIntegrityError, match="missing"):
            apply_missing_policy(df, ["ego_speed_mps", "nonexistent_feature"])
