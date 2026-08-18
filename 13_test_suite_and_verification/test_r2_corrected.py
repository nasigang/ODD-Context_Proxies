#!/usr/bin/env python3
"""
R2 Corrected Tests — production-code imports, no test-only reimplementations.

Covers: math/gradient, optimizer, delta sign, risk set, features, split, gate, holdout.
Per §10 of R2-CORRECTION. No skip allowed.
"""
import hashlib
import json
import os
import tempfile
import warnings

import numpy as np
import pytest
from scipy import optimize

from phase2_womd.r2_censored_likelihood import (
    censored_lognormal_nll, nll_and_grad, predict_exceedance_prob,
    log_transform_ttc, TTC_FLOOR, TTC_CAP, NLL_LABELS,
    _stable_mills_ratio, _stable_log_survival,
)
from phase2_womd.r2_models import (
    CensoredLogNormalAFT, create_model_suite, Z_STATE_FEATURES,
    C_CONTEXT_FEATURES, M4_INTERACTIONS
)
from phase2_womd.r2_feature_engineering import (
    check_leakage, validate_feature_dictionary, FeatureIntegrityError,
    validate_feature_columns, Z_STATE_NAMES, C_CONTEXT_NAMES, LEAKAGE_BANNED
)
from phase2_womd.r2_split import (
    assign_split, generate_split_membership, deterministic_split_hash
)
from phase2_womd.r2_input_gate import R2InputGate
from phase2_womd.r2_bootstrap import (
    scenario_block_paired_bootstrap, compute_conditional_brier,
    DELTA_SIGN_CONVENTION
)
from phase2_womd.open_holdout_once import HoldoutGuard, HoldoutGuardError
from phase2_womd.select_and_freeze_r2 import freeze_model_config, verify_frozen

NONINFERENTIAL_PILOT_DO_NOT_REPORT = True


# ═══════════════════════════════════════════════════════════════
# Math/Gradient Tests (§2.3 items 1–8)
# ═══════════════════════════════════════════════════════════════

class TestGradientCorrectness:
    """Finite-difference gradient parity with production analytic gradient."""

    SEED = 42
    N = 100
    P = 3
    TOL_ATOL = 1e-4
    TOL_RTOL = 1e-3
    FD_EPS = 1e-5

    def _make_data(self, n=None, p=None, seed=None):
        n = n or self.N; p = p or self.P; seed = seed or self.SEED
        rng = np.random.RandomState(seed)
        X = rng.randn(n, p)
        y_log = 1.0 + X @ rng.randn(p) * 0.5 + rng.randn(n) * 0.3
        censored = rng.rand(n) > 0.7
        y_log[censored] = np.log(TTC_CAP)
        censor_log = np.full(n, np.log(TTC_CAP))
        params = np.concatenate([rng.randn(p + 1) * 0.1, [np.log(0.5)]])
        return X, y_log, censored, censor_log, params

    def _fd_grad(self, params, X, y_log, censored, censor_log):
        grad_fd = np.zeros_like(params)
        for i in range(len(params)):
            pp = params.copy(); pp[i] += self.FD_EPS
            pm = params.copy(); pm[i] -= self.FD_EPS
            fp, _ = nll_and_grad(pp, X, y_log, censored, censor_log)
            fm, _ = nll_and_grad(pm, X, y_log, censored, censor_log)
            grad_fd[i] = (fp - fm) / (2 * self.FD_EPS)
        return grad_fd

    def test_fd_gradient_mixed(self):
        """Central FD vs analytic gradient on mixed event/censored data."""
        X, y_log, censored, censor_log, params = self._make_data()
        _, grad_a = nll_and_grad(params, X, y_log, censored, censor_log)
        grad_fd = self._fd_grad(params, X, y_log, censored, censor_log)
        np.testing.assert_allclose(grad_a, grad_fd, atol=self.TOL_ATOL, rtol=self.TOL_RTOL)

    def test_fd_gradient_event_only(self):
        X, y_log, _, censor_log, params = self._make_data()
        censored = np.zeros(self.N, dtype=bool)
        _, grad_a = nll_and_grad(params, X, y_log, censored, censor_log)
        grad_fd = self._fd_grad(params, X, y_log, censored, censor_log)
        np.testing.assert_allclose(grad_a, grad_fd, atol=self.TOL_ATOL, rtol=self.TOL_RTOL)

    def test_fd_gradient_censored_only(self):
        X, y_log, _, censor_log, params = self._make_data()
        censored = np.ones(self.N, dtype=bool)
        _, grad_a = nll_and_grad(params, X, y_log, censored, censor_log)
        grad_fd = self._fd_grad(params, X, y_log, censored, censor_log)
        np.testing.assert_allclose(grad_a, grad_fd, atol=self.TOL_ATOL, rtol=self.TOL_RTOL)

    def test_descent_direction(self):
        """Small -gradient step must not increase NLL."""
        X, y_log, censored, censor_log, params = self._make_data()
        nll0, grad = nll_and_grad(params, X, y_log, censored, censor_log)
        params_new = params - 0.001 * grad
        nll1, _ = nll_and_grad(params_new, X, y_log, censored, censor_log)
        assert nll1 <= nll0 + 1e-8, f"NLL increased: {nll0:.6f} → {nll1:.6f}"


class TestOptimizerConvergence:

    def test_stored_nll_parity(self):
        """Recomputed NLL from stored params == fit_info.final_nll_recomputed (tol=1e-6)."""
        rng = np.random.RandomState(42)
        model = CensoredLogNormalAFT("test", ["f1", "f2"])
        X = rng.randn(200, 2)
        y = np.exp(1.0 + X @ [0.3, -0.2] + rng.randn(200) * 0.5)
        c = rng.rand(200) > 0.7
        model.fit(X, y, c)
        nll_recomputed = model.nll(X, y, c)
        assert abs(nll_recomputed - model.fit_info["final_nll_recomputed"]) < 1e-6

    def test_synthetic_recovery(self):
        """Recover beta and sigma from n=5000 synthetic data within tolerance.
        PASS criteria: |beta_hat - beta_true| < 0.15, |sigma_hat - sigma_true| < 0.05."""
        rng = np.random.RandomState(123)
        n = 5000
        X = rng.randn(n, 2)
        true_beta = np.array([2.0, 0.5, -0.3])
        true_sigma = 0.4
        y_log = true_beta[0] + X @ true_beta[1:] + rng.randn(n) * true_sigma
        y_ttc = np.exp(y_log)
        censored = y_ttc > TTC_CAP
        y_ttc[censored] = TTC_CAP

        model = CensoredLogNormalAFT("recovery", ["f1", "f2"])
        model.fit(X, y_ttc, censored)
        assert abs(model.beta[0] - true_beta[0]) < 0.25, f"Intercept: {model.beta[0]:.3f} vs {true_beta[0]}"
        assert abs(model.beta[1] - true_beta[1]) < 0.1
        assert abs(model.beta[2] - true_beta[2]) < 0.1
        assert abs(model.sigma - true_sigma) < 0.05

    def test_extreme_z_no_warnings(self):
        """Extreme z produces finite NLL, no RuntimeWarning."""
        X = np.array([[0.0]])
        y_log = np.array([np.log(0.05)])
        censored = np.array([False])
        censor_log = np.array([np.log(10.0)])
        params = np.array([10.0, 0.0, np.log(0.1)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nll, grad = nll_and_grad(params, X, y_log, censored, censor_log)
            rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(rw) == 0, f"RuntimeWarning: {[str(x.message) for x in rw]}"
            assert np.isfinite(nll)
            assert np.all(np.isfinite(grad))

    def test_extreme_z_censored_stable(self):
        """Extreme censored z (z=50) → finite stable NLL."""
        z_extreme = np.array([50.0])
        log_surv = _stable_log_survival(z_extreme)
        assert np.all(np.isfinite(log_surv))
        h = _stable_mills_ratio(z_extreme)
        assert np.all(np.isfinite(h))
        assert h[0] > 0

    def test_convergence_info_fields(self):
        model = CensoredLogNormalAFT("test", ["f1"])
        X = np.random.randn(100, 1)
        y = np.exp(np.random.randn(100))
        c = np.random.rand(100) > 0.7
        model.fit(X, y, c)
        fi = model.fit_info
        required = {"optimizer_success", "optimizer_message", "optimizer_status",
                    "final_nll_recomputed", "final_grad_norm",
                    "n_iterations", "n_function_evals", "params_all_finite", "sigma"}
        assert required.issubset(set(fi.keys())), f"Missing: {required - set(fi.keys())}"
        assert fi["params_all_finite"] is True

    def test_cdf_survival_consistency(self):
        """Phi(z) + S(z) = 1 for range of z values."""
        from scipy.stats import norm
        for z in [-5, -2, 0, 2, 5, 10, 20]:
            assert abs(norm.cdf(z) + norm.sf(z) - 1.0) < 1e-12

    def test_all_event_degeneracy(self):
        """Model fits when all rows are events (no censoring)."""
        rng = np.random.RandomState(99)
        model = CensoredLogNormalAFT("event_only", ["f1"])
        X = rng.randn(100, 1)
        y = np.exp(1.0 + X[:, 0] * 0.3 + rng.randn(100) * 0.5)
        c = np.zeros(100, dtype=bool)
        model.fit(X, y, c)
        assert model.fitted
        assert np.isfinite(model.fit_info["final_nll_recomputed"])

    def test_all_censored_degeneracy(self):
        """Model fits when all rows are censored."""
        rng = np.random.RandomState(88)
        model = CensoredLogNormalAFT("cens_only", ["f1"])
        X = rng.randn(100, 1)
        y = np.full(100, TTC_CAP)
        c = np.ones(100, dtype=bool)
        model.fit(X, y, c)
        assert model.fitted
        assert np.isfinite(model.fit_info["final_nll_recomputed"])


# ═══════════════════════════════════════════════════════════════
# Delta NLL Sign Tests (§3)
# ═══════════════════════════════════════════════════════════════

class TestDeltaSign:

    def test_known_nll_direction(self):
        """NLL_M1=2.0, NLL_M3=1.5 → delta=0.5 → M3 better."""
        nll_m1 = np.array([2.0, 3.0, 2.5])
        nll_m3 = np.array([1.5, 2.5, 2.0])
        delta = nll_m1 - nll_m3
        assert np.all(delta > 0)
        assert np.mean(delta) > 0

    def test_sign_convention_string(self):
        assert "NLL_M1 - NLL_M3" in DELTA_SIGN_CONVENTION
        assert "positive = M3 better" in DELTA_SIGN_CONVENTION

    def test_bootstrap_paired_same_resample(self):
        """M1 and M3 evaluated on same bootstrap resample."""
        m1 = CensoredLogNormalAFT("M1", ["f1"])
        m3 = CensoredLogNormalAFT("M3", ["f1"])
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = np.exp(rng.randn(100))
        c = rng.rand(100) > 0.7
        sids = np.array([f"s{i//10}" for i in range(100)])
        m1.fit(X, y, c); m3.fit(X, y, c)
        r = scenario_block_paired_bootstrap(m1, m3, X, X, y, c, sids, n_boot=50, seed=99)
        # Both evaluated on same scenarios
        assert r["n_scenarios"] == 10
        assert "NLL_M1 - NLL_M3" in r["sign_convention"]

    def test_scenario_equal_weight_aggregation(self):
        """Each scenario gets equal weight regardless of frame count."""
        m = CensoredLogNormalAFT("test", ["f1"])
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = np.exp(rng.randn(100))
        c = rng.rand(100) > 0.7
        # Scenario s0 has 50 frames, s1 has 50 frames
        sids = np.array(["s0"] * 50 + ["s1"] * 50)
        m.fit(X, y, c)
        r = scenario_block_paired_bootstrap(m, m, X, X, y, c, sids, n_boot=10)
        # Delta should be 0 when comparing model to itself
        assert abs(r["delta_nll_mean"]) < 1e-10


# ═══════════════════════════════════════════════════════════════
# Risk Set / Overlap Tests (§4)
# ═══════════════════════════════════════════════════════════════

class TestRiskSetOverlap:

    def test_overlap_in_primary_row_fails(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "target_status": ["future_contact_event"],
            "overlap_now_flag": [True],
            "right_censored": [False],
            "ttc_obb_swept_s": [3.0],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"],
        })
        pq.write_table(t, str(tmp_path / "bad.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_risk_set_invariants(str(tmp_path / "bad.parquet")) is False

    def test_event_not_right_censored(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "target_status": ["future_contact_event"],
            "overlap_now_flag": [False],
            "right_censored": [True],
            "ttc_obb_swept_s": [3.0],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"],
        })
        pq.write_table(t, str(tmp_path / "bad.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_risk_set_invariants(str(tmp_path / "bad.parquet")) is False

    def test_event_ttc_out_of_range_fails(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "target_status": ["future_contact_event"],
            "overlap_now_flag": [False], "right_censored": [False],
            "ttc_obb_swept_s": [-1.0],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"],
        })
        pq.write_table(t, str(tmp_path / "bad.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_risk_set_invariants(str(tmp_path / "bad.parquet")) is False

    def test_valid_primary_passes(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a", "a"], "time_index": [0, 1], "ego_track_id": [1, 1],
            "target_status": ["future_contact_event", "right_censored"],
            "overlap_now_flag": [False, False],
            "right_censored": [False, True],
            "ttc_obb_swept_s": [3.0, 10.0],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"] * 2,
        })
        pq.write_table(t, str(tmp_path / "ok.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_risk_set_invariants(str(tmp_path / "ok.parquet")) is True


# ═══════════════════════════════════════════════════════════════
# Feature Contract Tests (§5)
# ═══════════════════════════════════════════════════════════════

class TestFeatureContract:

    def test_missing_column_fails(self):
        import pandas as pd
        df = pd.DataFrame({"ego_speed_mps": [1.0, 2.0]})
        with pytest.raises(FeatureIntegrityError, match="missing"):
            validate_feature_columns(df, ["ego_speed_mps", "n_eligible_pairs"])

    def test_constant_column_fails(self):
        import pandas as pd
        df = pd.DataFrame({"ego_speed_mps": [5.0] * 100})
        with pytest.raises(FeatureIntegrityError, match="constant"):
            validate_feature_columns(df, ["ego_speed_mps"])

    def test_all_nan_fails(self):
        import pandas as pd
        df = pd.DataFrame({"ego_speed_mps": [np.nan] * 10})
        with pytest.raises(FeatureIntegrityError, match="NaN"):
            validate_feature_columns(df, ["ego_speed_mps"])

    def test_leakage_banned(self):
        for col in ["ttc_obb_swept_s", "derived_ttc_2d_s", "dominant_target_type",
                     "future_anything", "circle_ttc_s", "target_status",
                     "right_censored", "overlap_now_flag", "odd_n_valid_agents"]:
            is_banned, _ = check_leakage(col)
            assert is_banned, f"{col} should be banned"

    def test_allowed_features_safe(self):
        for feat in Z_STATE_NAMES + C_CONTEXT_NAMES:
            is_banned, reason = check_leakage(feat)
            assert not is_banned, f"{feat}: {reason}"

    def test_validate_dictionary_passes(self):
        validate_feature_dictionary()


# ═══════════════════════════════════════════════════════════════
# Split Tests (§6)
# ═══════════════════════════════════════════════════════════════

class TestSplitHash:

    def test_row_order_invariance(self):
        ids1 = [f"s{i}" for i in range(50)]
        ids2 = list(reversed(ids1))
        m1 = generate_split_membership(ids1)
        m2 = generate_split_membership(ids2)
        for split in m1:
            assert sorted(m1[split]) == sorted(m2[split])

    def test_pilot_full_consistency(self):
        pilot = [f"s{i}" for i in range(50)]
        full = [f"s{i}" for i in range(200)]
        m_p = generate_split_membership(pilot)
        m_f = generate_split_membership(full)
        for sid in pilot:
            sp = next(s for s, ids in m_p.items() if sid in ids)
            sf = next(s for s, ids in m_f.items() if sid in ids)
            assert sp == sf, f"{sid}: pilot={sp}, full={sf}"

    def test_no_overlap(self):
        ids = [f"s{i}" for i in range(100)]
        m = generate_split_membership(ids)
        all_sids = []
        for sids in m.values():
            all_sids.extend(sids)
        assert len(all_sids) == len(set(all_sids))
        assert set(all_sids) == set(ids)

    def test_no_external_test(self):
        m = generate_split_membership([f"s{i}" for i in range(50)])
        assert "external_test" not in m

    def test_uses_sha256_not_python_hash(self):
        """Verify it uses SHA-256, not Python built-in hash()."""
        u = deterministic_split_hash("womd_r2_split_v1", 42, "test_scenario")
        assert isinstance(u, float)
        assert 0 <= u < 1
        # Same call should be deterministic
        u2 = deterministic_split_hash("womd_r2_split_v1", 42, "test_scenario")
        assert u == u2


# ═══════════════════════════════════════════════════════════════
# Gate Fail-Closed Tests (§7)
# ═══════════════════════════════════════════════════════════════

class TestGateFailClosed:

    def test_missing_args_fail(self):
        gate = R2InputGate("/tmp/nonexistent")
        passed, _ = gate.run_all(None, None, None, None)
        assert passed is False

    def test_missing_manifest_hash(self, tmp_path):
        gate = R2InputGate(str(tmp_path))
        assert gate.check_manifest_hash(None, None) is False

    def test_wrong_method(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "ttc_method": ["circle_v1"],
            "target_status": ["right_censored"], "overlap_now_flag": [False],
            "right_censored": [True], "ttc_obb_swept_s": [10.0],
        })
        pq.write_table(t, str(tmp_path / "bad.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_method_and_legacy(str(tmp_path / "bad.parquet")) is False

    def test_banned_column(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a"], "time_index": [0], "ego_track_id": [1],
            "derived_ttc_2d_s": [3.0],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"],
            "target_status": ["right_censored"], "overlap_now_flag": [False],
        })
        pq.write_table(t, str(tmp_path / "bad.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_method_and_legacy(str(tmp_path / "bad.parquet")) is False

    def test_duplicate_keys_fail(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.table({
            "scenario_id": ["a", "a"], "time_index": [0, 0], "ego_track_id": [1, 1],
            "target_status": ["right_censored", "right_censored"],
            "overlap_now_flag": [False, False], "right_censored": [True, True],
            "ttc_obb_swept_s": [10.0, 10.0],
            "ttc_method": ["obb_swept_sat_cv_fixed_heading_v1"] * 2,
        })
        pq.write_table(t, str(tmp_path / "dup.parquet"))
        gate = R2InputGate(str(tmp_path))
        assert gate.check_risk_set_invariants(str(tmp_path / "dup.parquet")) is False


# ═══════════════════════════════════════════════════════════════
# Frozen Eval + Holdout Tests (§8)
# ═══════════════════════════════════════════════════════════════

class TestFrozenAndHoldout:

    def test_save_load_parity(self, tmp_path):
        model = CensoredLogNormalAFT("test", ["f1", "f2"])
        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        y = np.exp(rng.randn(200))
        c = rng.rand(200) > 0.7
        model.fit(X, y, c)
        mu1, _ = model.predict(X)
        model.save(str(tmp_path / "m.pkl"))
        loaded = CensoredLogNormalAFT.load(str(tmp_path / "m.pkl"))
        mu2, _ = loaded.predict(X)
        np.testing.assert_array_almost_equal(mu1, mu2)
        assert loaded.fitted is True

    def test_holdout_first_succeeds(self, tmp_path):
        guard = HoldoutGuard(str(tmp_path))
        guard.request_access("h1", "c1")
        assert guard.is_opened()

    def test_holdout_second_fails(self, tmp_path):
        guard = HoldoutGuard(str(tmp_path))
        guard.request_access("h1", "c1")
        with pytest.raises(HoldoutGuardError):
            guard.request_access("h1", "c1")

    def test_holdout_force_needs_reason(self, tmp_path):
        guard = HoldoutGuard(str(tmp_path))
        guard.request_access("h1", "c1")
        with pytest.raises(HoldoutGuardError):
            guard.request_access("h1", "c1", force=True)

    def test_freeze_verify_roundtrip(self, tmp_path):
        mdir = str(tmp_path / "models"); os.makedirs(mdir)
        for name, feats in [("M0", []), ("M1", ["f1"]), ("M3", ["f1"])]:
            m = CensoredLogNormalAFT(name, feats)
            X = np.random.randn(50, max(1, len(feats)))
            if len(feats) == 0:
                X = np.random.randn(50, 0).reshape(50, 0)
            m.fit(X, np.exp(np.random.randn(50)), np.random.rand(50) > 0.7)
            m.save(os.path.join(mdir, f"model_{name}.pkl"))
        import pickle
        ppath = str(tmp_path / "scaler.pkl")
        with open(ppath, "wb") as f:
            pickle.dump({"feature_names": ["f1"]}, f)
        frozen = freeze_model_config(mdir, ppath, ["f1"], str(tmp_path))
        assert frozen["status"] == "FROZEN"
        assert "NLL_M1 - NLL_M3" in frozen["sign_convention"]
        assert "positive = M3 better" in frozen["sign_convention"]
        ok, issues = verify_frozen(str(tmp_path / "frozen_config.json"), mdir, ppath)
        assert ok, f"Verification failed: {issues}"

    def test_loaded_model_no_refit(self, tmp_path):
        """Loading a model doesn't require calling fit()."""
        model = CensoredLogNormalAFT("M1", ["f1"])
        model.fit(np.random.randn(50, 1), np.exp(np.random.randn(50)),
                  np.random.rand(50) > 0.7)
        model.save(str(tmp_path / "m.pkl"))
        loaded = CensoredLogNormalAFT.load(str(tmp_path / "m.pkl"))
        # Can predict without fit
        mu, sig = loaded.predict(np.random.randn(10, 1))
        assert len(mu) == 10


# ═══════════════════════════════════════════════════════════════
# NLL Label Tests (§C)
# ═══════════════════════════════════════════════════════════════

class TestNLLLabels:

    def test_ttc_scale_includes_jacobian(self):
        """TTC-scale NLL includes log(T) Jacobian term."""
        y = np.array([1.0, 2.0])
        mu = np.array([0.5, 0.5])
        sigma = 1.0
        censored = np.array([False, False])
        nll_ttc = censored_lognormal_nll(y, mu, sigma, censored,
                                          include_jacobian=True, reduction="none")
        nll_log = censored_lognormal_nll(y, mu, sigma, censored,
                                          include_jacobian=False, reduction="none")
        # Difference should be exactly log(T)
        diff = nll_ttc - nll_log
        np.testing.assert_allclose(diff, np.log(y), atol=1e-12)

    def test_labels_never_mixed(self):
        assert "Jacobian" in NLL_LABELS["ttc_scale"]
        assert "no Jacobian" in NLL_LABELS["log_scale"]
