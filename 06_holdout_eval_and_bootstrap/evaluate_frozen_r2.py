#!/usr/bin/env python3
"""
Evaluate Frozen R2 Model — loads saved artifacts, evaluates WITHOUT refitting.

Corrections from R2-CORRECTION:
- Imports CensoredLogNormalAFT (not deleted CensoredGaussianModel)
- Uses censored_nll_components (not deleted censored_gaussian_nll_components)
- Split allowlist: train only (Stage A); +internal_val (Stage B); holdout/external always blocked
- Primary delta: NLL_M1 - NLL_M3 (positive = M3 better)
- censor_time_s passed to NLL/bootstrap/calibration
- Brier FAIL if censor_time < tau (no numeric + IPCW_REQUIRED)
"""
import hashlib
import json
import os

import numpy as np

from phase2_womd.r2_models import CensoredLogNormalAFT
from phase2_womd.r2_censored_likelihood import (
    censored_lognormal_nll, censored_nll_components, predict_exceedance_prob,
    NLL_LABELS
)
from phase2_womd.r2_bootstrap import (
    scenario_block_paired_bootstrap, compute_conditional_brier,
    DELTA_SIGN_CONVENTION
)


SENSITIVITY_TAUS = [1.0, 2.0, 3.0, 5.0]
BRIDGE_TAU = 3.220298

# Split allowlist — general evaluator NEVER allows holdout/external
ALLOWED_SPLITS_STAGE_A = {"train"}
ALLOWED_SPLITS_STAGE_B = {"train", "internal_val"}
BLOCKED_SPLITS = {"internal_holdout", "external_test"}


class SplitAccessError(Exception):
    pass


class FrozenEvaluator:
    """Evaluates frozen models without refitting."""

    def __init__(self, model_dir, preproc_path, stage="A"):
        self.model_dir = model_dir
        self.preproc_path = preproc_path
        self.stage = stage
        self._models = {}
        self._preproc = None
        self._allowed = ALLOWED_SPLITS_STAGE_A if stage == "A" else ALLOWED_SPLITS_STAGE_B

    def _check_split(self, split_name):
        if split_name in BLOCKED_SPLITS:
            raise SplitAccessError(
                f"Split '{split_name}' is BLOCKED in general evaluator. "
                f"Use holdout command (open_holdout_once.py) for holdout.")
        if split_name not in self._allowed:
            raise SplitAccessError(
                f"Split '{split_name}' not allowed in Stage {self.stage}. "
                f"Allowed: {self._allowed}")

    def load_model(self, name):
        """Load a frozen model. Does NOT call fit()."""
        path = os.path.join(self.model_dir, f"model_{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")
        model = CensoredLogNormalAFT.load(path)
        assert model.fitted, f"Loaded model {name} is not fitted!"
        with open(path, "rb") as f:
            model._artifact_hash = hashlib.sha256(f.read()).hexdigest()
        self._models[name] = model
        return model

    def load_preprocessing(self):
        """Load frozen preprocessing artifacts."""
        import pickle
        with open(self.preproc_path, "rb") as f:
            self._preproc = pickle.load(f)
        return self._preproc

    def evaluate_on_split(self, X, y_ttc, censored, scenario_ids, split_name,
                          censor_time=None, model_names=None):
        """
        Evaluate all loaded models on a data split.
        Does NOT call .fit(). Enforces split allowlist.
        """
        self._check_split(split_name)

        if model_names is None:
            model_names = list(self._models.keys())

        report = {
            "split": split_name,
            "nll_label": NLL_LABELS["ttc_scale"],
            "n_frames": len(y_ttc),
            "n_scenarios": len(np.unique(scenario_ids)),
            "n_events": int((~censored).sum()),
            "n_censored": int(censored.sum()),
            "models": {},
            "FIT_CALLED_DURING_EVAL": False,
        }

        for name in model_names:
            model = self._models[name]
            X_model = self._select_features(X, model)
            mu, sigma = model.predict(X_model)

            # NLL with row-level censor_time
            comps = censored_nll_components(
                y_ttc, mu, sigma, censored,
                include_jacobian=True,
                censor_time=censor_time)

            # Conditional Brier at each tau
            brier_results = {}
            all_taus = SENSITIVITY_TAUS + [BRIDGE_TAU]
            for tau in all_taus:
                brier = compute_conditional_brier(
                    model, X_model, y_ttc, censored, tau,
                    censor_time=censor_time)

                # Enforce: no numeric Brier if censor_time < tau
                if not brier.get("censor_time_ge_tau", True):
                    brier["brier_score"] = None
                    brier["FAIL"] = "censor_time < tau for some rows; naive Brier invalid"

                brier_results[f"tau_{tau:.3f}"] = brier

            report["models"][name] = {
                "nll": comps,
                "conditional_brier": brier_results,
                "artifact_hash": getattr(model, "_artifact_hash", "unknown"),
                "parameter_count": model.fit_info.get("parameter_count", 0),
                "sigma": model.fit_info.get("sigma"),
            }

        return report

    def run_primary_comparison(self, X, y_ttc, censored, scenario_ids,
                               censor_time=None,
                               n_boot=1000, seed=42):
        """
        Primary comparison: Delta = NLL_M1 - NLL_M3 (positive = M3 better).

        model_a = M1, model_b = M3 (LOCKED order).
        """
        if "M1" not in self._models or "M3" not in self._models:
            raise ValueError("Both M1 and M3 must be loaded for primary comparison")

        model_a = self._models["M1"]  # a = M1
        model_b = self._models["M3"]  # b = M3
        X_a = self._select_features(X, model_a)
        X_b = self._select_features(X, model_b)

        result = scenario_block_paired_bootstrap(
            model_a, model_b, X_a, X_b,
            y_ttc, censored, scenario_ids,
            n_boot=n_boot, seed=seed,
            censor_time=censor_time,
        )

        result["primary_comparison"] = "M3 vs M1"
        result["sign_convention"] = DELTA_SIGN_CONVENTION
        result["nll_label"] = NLL_LABELS["ttc_scale"]

        return result

    def _select_features(self, X_full, model):
        """Select feature columns for a model. Does NOT mutate X_full."""
        from phase2_womd.r2_feature_engineering import Z_STATE_NAMES, C_CONTEXT_NAMES
        all_base = Z_STATE_NAMES + C_CONTEXT_NAMES
        if model.n_features == 0:
            return np.zeros((X_full.shape[0], 0))
        col_idx = []
        extra_cols = []
        for fn in model.feature_names:
            if fn in all_base:
                col_idx.append(all_base.index(fn))
            elif "__x__" in fn:
                parts = fn.split("__x__")
                if parts[0] in all_base and parts[1] in all_base:
                    i1, i2 = all_base.index(parts[0]), all_base.index(parts[1])
                    extra_cols.append(X_full[:, i1] * X_full[:, i2])
        X_model = X_full[:, col_idx] if col_idx else np.zeros((len(X_full), 0))
        if extra_cols:
            X_model = np.column_stack([X_model] + extra_cols)
        return X_model

    def save_report(self, report, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
