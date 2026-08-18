#!/usr/bin/env python3
"""
R2 Models — M0–M4 homoscedastic censored log-normal AFT.

log(T_i) = mu_i + sigma * epsilon_i,  epsilon ~ N(0,1)
mu_i = beta_0 + X_i @ beta
sigma > 0, common across rows within a model.

Optimizer: scipy.optimize.minimize(method="L-BFGS-B").
"""
import hashlib
import json
import os
import pickle

import numpy as np
from scipy.optimize import minimize

from phase2_womd.r2_censored_likelihood import (
    nll_and_grad, censored_lognormal_nll, predict_exceedance_prob,
    log_transform_ttc, TTC_FLOOR, TTC_CAP, NLL_LABELS
)

# Predeclared M4 interactions (fixed BEFORE seeing any results)
M4_INTERACTIONS = [
    ("ego_speed_mps", "n_eligible_pairs"),
    ("ego_speed_mps", "max_closing_speed_mps"),
    ("n_eligible_pairs", "traffic_n_valid_agents"),
]

Z_STATE_FEATURES = [
    "ego_speed_mps", "ego_accel_mps2", "ego_yaw_rate_rps",
    "n_eligible_pairs", "min_pair_distance_m", "max_closing_speed_mps",
]

C_CONTEXT_FEATURES = [
    "traffic_n_valid_agents",
]


class CensoredLogNormalAFT:
    """
    Homoscedastic censored log-normal AFT model.

    Parameters: [beta_0, beta_1, ..., beta_p, log_sigma]
    """

    def __init__(self, name, feature_names, model_id=None):
        self.name = name
        self.feature_names = list(feature_names)
        self.model_id = model_id or name
        self.n_features = len(self.feature_names)
        self.params = None
        self.fitted = False
        self.fit_info = {}

    @property
    def beta(self):
        if self.params is None:
            return None
        return self.params[:self.n_features + 1]

    @property
    def log_sigma(self):
        if self.params is None:
            return None
        return self.params[self.n_features + 1]

    @property
    def sigma(self):
        if self.log_sigma is None:
            return None
        return np.exp(self.log_sigma)

    def _objective(self, params, X, y_log, censored, censor_log):
        """Objective wrapper for scipy."""
        return nll_and_grad(params, X, y_log, censored, censor_log)

    def fit(self, X, y_ttc, censored, censor_time=None, maxiter=500):
        """
        Fit via scipy L-BFGS-B.

        Parameters
        ----------
        X : (n, p) feature matrix (NO intercept column)
        y_ttc : (n,) TTC in seconds
        censored : (n,) bool
        censor_time : (n,) or None (defaults to TTC_CAP)
        """
        X = np.asarray(X, dtype=np.float64)
        y_ttc = np.asarray(y_ttc, dtype=np.float64)
        censored = np.asarray(censored, dtype=bool)
        n, p = X.shape

        assert p == self.n_features, f"Expected {self.n_features} features, got {p}"

        y_log = log_transform_ttc(y_ttc)
        if censor_time is not None:
            censor_log = log_transform_ttc(np.asarray(censor_time, dtype=np.float64))
        else:
            censor_log = np.full(n, np.log(TTC_CAP))

        # Initial parameters: OLS on events for beta, log(std(y_log)) for log_sigma
        x0 = np.zeros(p + 2)
        event_mask = ~censored
        if event_mask.sum() > p + 1:
            Xe = np.column_stack([np.ones(event_mask.sum()), X[event_mask]])
            try:
                beta_init = np.linalg.lstsq(Xe, y_log[event_mask], rcond=None)[0]
                x0[:p + 1] = beta_init
            except np.linalg.LinAlgError:
                pass
            resid = y_log[event_mask] - Xe @ x0[:p + 1]
            x0[p + 1] = np.log(max(np.std(resid), 0.1))
        else:
            x0[p + 1] = np.log(1.0)

        # Bounds: log_sigma > log(0.01)
        bounds = [(None, None)] * (p + 1) + [(np.log(0.01), np.log(100.0))]

        result = minimize(
            fun=lambda params: self._objective(params, X, y_log, censored, censor_log),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )

        self.params = result.x

        # Recompute final NLL from stored params
        final_nll, final_grad = nll_and_grad(self.params, X, y_log, censored, censor_log)

        # Check finite
        if not np.all(np.isfinite(self.params)):
            raise ValueError(f"Non-finite parameters after fit: {self.params}")

        self.fitted = True
        self.fit_info = {
            "n_samples": n,
            "n_events": int(event_mask.sum()),
            "n_censored": int(censored.sum()),
            "n_features": self.n_features,
            "parameter_count": len(self.params),
            "sigma": float(self.sigma),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_status": int(result.status),
            "final_nll_recomputed": float(final_nll),
            "final_grad_norm": float(np.linalg.norm(final_grad)),
            "n_iterations": int(result.nit),
            "n_function_evals": int(result.nfev),
            "params_all_finite": bool(np.all(np.isfinite(self.params))),
            "nll_label": NLL_LABELS["ttc_scale"],
        }

    def predict_mu(self, X):
        """Predict mu = beta_0 + X @ beta[1:]."""
        assert self.fitted, "Model not fitted"
        X = np.asarray(X, dtype=np.float64)
        return self.beta[0] + X @ self.beta[1:]

    def predict(self, X):
        """Return (mu, sigma)."""
        return self.predict_mu(X), self.sigma

    def nll(self, X, y_ttc, censored, censor_time=None):
        """Compute NLL on given data using stored parameters."""
        mu = self.predict_mu(X)
        return censored_lognormal_nll(y_ttc, mu, self.sigma, censored,
                                      censor_time=censor_time)

    def predict_exceedance(self, X, tau):
        """P(TTC <= tau | E_primary=1, features). No p_exposure."""
        mu = self.predict_mu(X)
        return predict_exceedance_prob(mu, self.sigma, tau)

    def save(self, path):
        """Save to pickle. Returns SHA256 hash."""
        assert self.fitted, "Cannot save unfitted model"
        state = {
            "name": self.name, "model_id": self.model_id,
            "feature_names": self.feature_names,
            "params": self.params, "fit_info": self.fit_info,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def load(cls, path):
        """Load from pickle. Does NOT call fit()."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(state["name"], state["feature_names"], state.get("model_id"))
        model.params = state["params"]
        model.fit_info = state["fit_info"]
        model.fitted = True
        return model

    def artifact_hash(self, path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


def create_model_suite():
    """Create M0–M4 model specifications."""
    models = {}
    models["M0"] = CensoredLogNormalAFT("M0_unconditional", [])
    models["M1"] = CensoredLogNormalAFT("M1_z_state", Z_STATE_FEATURES)
    models["M2"] = CensoredLogNormalAFT("M2_c_context", C_CONTEXT_FEATURES)
    models["M3"] = CensoredLogNormalAFT("M3_z_c", Z_STATE_FEATURES + C_CONTEXT_FEATURES)
    m4_feats = list(Z_STATE_FEATURES + C_CONTEXT_FEATURES)
    for f1, f2 in M4_INTERACTIONS:
        m4_feats.append(f"{f1}__x__{f2}")
    models["M4"] = CensoredLogNormalAFT("M4_interactions", m4_feats)
    return models
