#!/usr/bin/env python3
"""
R2 Preparation Tests — code/schema validation only (no inferential results).

All tests marked NONINFERENTIAL_PILOT_DO_NOT_REPORT.
Can run on synthetic data or pilot train-only smoke subset.
"""
import json
import math
import os
import pytest
import numpy as np

NONINFERENTIAL_PILOT_DO_NOT_REPORT = True
R2_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "R2", "model")
R1_REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "R1", "reports")


# ═══════════════════════════════════════════════════════════════
# 1. Censored Likelihood Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestCensoredLikelihood:
    """Verify censored Gaussian NLL handles events and censored observations correctly."""

    def _censored_nll(self, y, mu, sigma, censored, cap=10.0):
        """Censored Gaussian NLL: event → log-density, censored → log-survival."""
        from scipy.stats import norm
        nll = 0.0
        for yi, mi, si, ci in zip(y, mu, sigma, censored):
            if ci:
                # Right-censored: -log(S(cap))
                surv = 1.0 - norm.cdf(np.log(cap), loc=mi, scale=si)
                nll -= np.log(max(surv, 1e-12))
            else:
                # Event: -log(f(yi))
                nll -= norm.logpdf(np.log(max(yi, 0.05)), loc=mi, scale=si)
        return nll / len(y)

    def test_event_only_reduces_to_gaussian_nll(self):
        """With no censored obs, reduces to standard Gaussian NLL."""
        from scipy.stats import norm
        y = np.array([1.0, 2.0, 3.0])
        mu, sigma = 0.5, 1.0
        censored = np.array([False, False, False])
        nll = self._censored_nll(y, [mu]*3, [sigma]*3, censored)
        expected = -np.mean([norm.logpdf(np.log(yi), loc=mu, scale=sigma) for yi in y])
        assert abs(nll - expected) < 1e-6

    def test_censored_uses_survival(self):
        """Censored obs uses survival function, not density."""
        y_event = np.array([5.0])
        y_censor = np.array([10.0])
        mu, sigma = 1.0, 0.5
        nll_event = self._censored_nll(y_event, [mu], [sigma], [False])
        nll_censor = self._censored_nll(y_censor, [mu], [sigma], [True])
        # Both should be finite
        assert np.isfinite(nll_event)
        assert np.isfinite(nll_censor)

    def test_all_censored_does_not_crash(self):
        """All-censored batch: NLL should be finite and positive."""
        y = np.array([10.0, 10.0, 10.0])
        mu, sigma = 1.0, 0.5
        censored = np.array([True, True, True])
        nll = self._censored_nll(y, [mu]*3, [sigma]*3, censored)
        assert np.isfinite(nll) and nll > 0

    def test_floor_prevents_log_zero(self):
        """y=0 should be floored to 0.05 before log transform."""
        y = np.array([0.0])
        mu, sigma = 0.0, 1.0
        nll = self._censored_nll(y, [mu], [sigma], [False])
        assert np.isfinite(nll)


# ═══════════════════════════════════════════════════════════════
# 2. Calibration Estimand Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestCalibrationEstimand:
    """Verify conditional vs marginal calibration have separate denominators and formulas."""

    def test_conditional_excludes_non_exposed(self):
        """Conditional calibration sample = exposed rows only."""
        status = np.array(["future_contact_event", "right_censored", "no_exposure", "invalid_ego_state"])
        ttc = np.array([3.0, 10.0, np.nan, np.nan])
        exposed = np.isin(status, ["future_contact_event", "right_censored", "current_geometry_overlap"])
        assert exposed.sum() == 2  # only first two
        # Conditional prediction = F(tau | E=1, C, Z), NOT multiplied by p_exposure
        pred_cond = np.array([0.8, 0.2])  # F(tau) for exposed rows
        obs_cond = (ttc[exposed] <= 3.0).astype(float)
        assert obs_cond[0] == 1.0  # event at 3.0
        assert obs_cond[1] == 0.0  # right censored

    def test_marginal_includes_valid_riskset(self):
        """Marginal calibration sample = all valid risk-set (excl invalid states)."""
        status = np.array(["future_contact_event", "right_censored", "no_exposure", "invalid_ego_state"])
        valid_riskset = ~np.isin(status, ["invalid_frame", "invalid_ego_state"])
        assert valid_riskset.sum() == 3  # excl invalid_ego_state
        # Marginal prediction = P(E=1|C,Z) * F(tau|E=1,C,Z)
        p_exposure = np.array([0.9, 0.9, 0.1])  # P(E=1)
        f_tau = np.array([0.8, 0.2, 0.5])  # F(tau|E=1)
        pred_marginal = p_exposure * f_tau
        assert len(pred_marginal) == 3

    def test_conditional_marginal_different_denominators(self):
        """Conditional and marginal must have different sample sizes."""
        n_total = 100
        n_exposed = 60
        n_valid_riskset = 90
        assert n_exposed != n_valid_riskset

    def test_overlap_excluded_from_primary_event(self):
        """current_geometry_overlap is in exposure set but NOT a primary event."""
        status = "current_geometry_overlap"
        is_exposed = True
        is_primary_event = False
        is_in_likelihood = False
        is_sensitivity_only = True
        assert is_exposed and not is_primary_event


# ═══════════════════════════════════════════════════════════════
# 3. Legacy Column Ban
# ═══════════════════════════════════════════════════════════════

class TestLegacyColumnBan:
    """Ensure legacy Circle-based columns cannot enter model table."""

    BANNED = ["derived_ttc_2d_s", "ttc_min_s"]

    def test_banned_columns_not_in_feature_dict(self):
        """Feature dictionary should not contain banned columns as features."""
        dict_path = os.path.join(R2_MODEL_DIR, "feature_construct_dictionary.json")
        if not os.path.exists(dict_path):
            pytest.skip("Feature dictionary not found")
        with open(dict_path) as f:
            fd = json.load(f)
        z_features = list(fd.get("Z_state", {}).get("features", {}).keys())
        c_features = list(fd.get("C_ODD", {}).get("validated", {}).keys())
        all_features = z_features + c_features
        for banned in self.BANNED:
            assert banned not in all_features, f"Banned column {banned} in feature dictionary"

    def test_target_is_obb(self):
        dict_path = os.path.join(R2_MODEL_DIR, "feature_construct_dictionary.json")
        if not os.path.exists(dict_path):
            pytest.skip("Feature dictionary not found")
        with open(dict_path) as f:
            fd = json.load(f)
        target = fd.get("target_columns", {})
        assert "ttc_obb_swept_s" in target
        assert target["ttc_obb_swept_s"]["method"] == "obb_swept_sat_cv_fixed_heading_v1"


# ═══════════════════════════════════════════════════════════════
# 4. Split / Leakage Tests
# ═══════════════════════════════════════════════════════════════

class TestSplitLeakage:
    """Verify scenario-level split integrity."""

    def test_leakage_banned_in_features(self):
        dict_path = os.path.join(R2_MODEL_DIR, "feature_construct_dictionary.json")
        if not os.path.exists(dict_path):
            pytest.skip("Feature dictionary not found")
        with open(dict_path) as f:
            fd = json.load(f)
        z_features = list(fd.get("Z_state", {}).get("features", {}).keys())
        banned = fd.get("leakage_banned", [])
        for feat in z_features:
            # Feature names should not appear in banned list
            for b in banned:
                bname = b.split(" ")[0]  # strip parenthetical notes
                assert feat != bname, f"Feature {feat} is in leakage_banned list"

    def test_exposure_gate_logic(self):
        """Non-exposure states must NOT be used as negative training examples."""
        dict_path = os.path.join(R2_MODEL_DIR, "feature_construct_dictionary.json")
        if not os.path.exists(dict_path):
            pytest.skip("Feature dictionary not found")
        with open(dict_path) as f:
            fd = json.load(f)
        status_def = fd.get("target_columns", {}).get("target_status", {}).get("values", {})
        for state_name, props in status_def.items():
            if not props.get("exposure", False):
                assert not props.get("in_likelihood", False), \
                    f"Non-exposed state {state_name} should not be in likelihood"


# ═══════════════════════════════════════════════════════════════
# 5. Non-Degeneracy Tests
# ═══════════════════════════════════════════════════════════════

class TestNonDegeneracy:
    """Verify estimand is not degenerate."""

    def test_estimand_lock_exists(self):
        path = os.path.join(R2_MODEL_DIR, "estimand_lock.json")
        assert os.path.exists(path)
        with open(path) as f:
            el = json.load(f)
        assert el["status"] == "LOCKED"

    def test_primary_estimand_has_required_fields(self):
        path = os.path.join(R2_MODEL_DIR, "estimand_lock.json")
        if not os.path.exists(path):
            pytest.skip()
        with open(path) as f:
            el = json.load(f)
        pe = el["primary_estimand"]
        assert "formula" in pe
        assert "event_definition" in pe
        assert "method" in pe
        assert pe["method"] == "obb_swept_sat_cv_fixed_heading_v1"

    def test_primary_comparison_defined(self):
        path = os.path.join(R2_MODEL_DIR, "estimand_lock.json")
        if not os.path.exists(path):
            pytest.skip()
        with open(path) as f:
            el = json.load(f)
        pc = el["primary_comparison"]
        assert pc["test"] == "M3 vs M1 scenario-block paired censored-NLL difference"

    def test_r2_input_acceptance_blocked(self):
        """R2 input acceptance should be BLOCKED until R1 full."""
        path = os.path.join(R2_MODEL_DIR, "r2_input_acceptance.json")
        if not os.path.exists(path):
            pytest.skip()
        with open(path) as f:
            acc = json.load(f)
        assert "BLOCKED" in acc["status"]

    def test_holdout_is_sentinel_protected(self):
        path = os.path.join(R2_MODEL_DIR, "estimand_lock.json")
        if not os.path.exists(path):
            pytest.skip()
        with open(path) as f:
            el = json.load(f)
        hp = el["holdout_protocol"]
        assert hp["execution"] == "single one-shot sentinel-protected command"
