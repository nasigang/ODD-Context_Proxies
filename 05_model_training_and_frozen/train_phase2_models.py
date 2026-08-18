#!/usr/bin/env python3
"""
Phase 2 Model Training
========================
Trains M0–M5 models on WOMD frame targets using y_log_ttc + censoring.

Models:
  M0  Unconditional hurdle baseline
  M1  C_ODD conditioned
  M2  C_ODD + Z_state
  M3  M2 + selected 2-way interaction terms
  M4  Separate gate-expert
  M5  M4 + warp (placeholder — NOT trained)

Gate-expert architecture:
  p_e(C,Z) = P(exposure=1 | C, Z)          — logistic gate
  F_θ(log(τ) | C, Z, exposure=1)           — Gaussian expert
  R_τ = p_e · F_θ(log(τ) | C, Z, E=1)     — risk score

Usage:
    python phase2_womd/train_phase2_models.py --max-scenarios 20
"""

import argparse
import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from phase2_womd.build_model_table import (
    C_ODD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMNS,
    Z_STATE_FEATURES,
    build_model_table,
)
from phase2_womd.obb_ttc import T_MAX_S, T_MIN_S

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "M0": {
        "name": "Unconditional Hurdle Baseline",
        "features": [],
        "description": "Marginal exposure rate + marginal log-TTC distribution",
    },
    "M1": {
        "name": "C_ODD Conditioned",
        "features": C_ODD_FEATURES,
        "description": "Logistic gate + Gaussian expert conditioned on ODD features",
    },
    "M2": {
        "name": "C_ODD + Z_state",
        "features": C_ODD_FEATURES + Z_STATE_FEATURES,
        "description": "Gate + expert conditioned on ODD + ego state",
    },
    "M3": {
        "name": "M2 + Interactions",
        "features": C_ODD_FEATURES + Z_STATE_FEATURES,
        "interactions": True,
        "description": "M2 plus selected 2-way C×Z interaction terms",
    },
    "M4": {
        "name": "Separate Gate-Expert",
        "features": C_ODD_FEATURES + Z_STATE_FEATURES,
        "separate_gate_expert": True,
        "description": "Separately trained gate and expert with risk composition",
    },
    "M5": {
        "name": "M4 + Warp (Placeholder)",
        "features": C_ODD_FEATURES + Z_STATE_FEATURES,
        "placeholder": True,
        "description": "Warp disabled at this stage — placeholder only",
    },
}


# ---------------------------------------------------------------------------
# Censored Gaussian log-likelihood
# ---------------------------------------------------------------------------

def censored_nll(
    y: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Compute negative log-likelihood for censored Gaussian.

    Args:
        y: observed log-TTC values
        delta: 1 if event, 0 if censored
        mu: predicted mean
        sigma: predicted std (>0)

    Returns:
        Average negative log-likelihood
    """
    eps = 1e-12
    sigma = np.maximum(sigma, eps)

    # Events: log φ((y-μ)/σ) - log σ
    ll_event = norm.logpdf(y, loc=mu, scale=sigma)

    # Censored: log S(y | μ, σ) = log(1 - Φ((y-μ)/σ))
    z = (y - mu) / sigma
    ll_censored = norm.logsf(z)   # log survival

    ll = np.where(delta == 1, ll_event, ll_censored)

    # Filter out invalid entries
    valid = np.isfinite(ll)
    if valid.sum() == 0:
        return float("inf")
    return -float(np.mean(ll[valid]))


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class HurdleModel:
    """Generic hurdle model: gate (exposure) + expert (log-TTC)."""

    def __init__(self, name: str):
        self.name = name
        self.gate = None       # P(exposure=1)
        self.mu = None         # E[log-TTC | exposure=1]
        self.sigma = None      # Std[log-TTC | exposure=1]
        self.gate_model = None  # fitted sklearn model (if conditional)
        self.config = {}

    def fit_unconditional(self, df_train: pd.DataFrame):
        """M0: marginal rates."""
        exposed = df_train["exposure_flag"].values
        self.gate = float(exposed.mean())

        event_mask = df_train["target_status"] == "event"
        y_events = df_train.loc[event_mask, "y_log_ttc"].dropna()

        if len(y_events) > 1:
            self.mu = float(y_events.mean())
            self.sigma = float(y_events.std())
        else:
            self.mu = float(math.log(T_MAX_S))
            self.sigma = 1.0

        self.config = {
            "type": "unconditional",
            "gate_rate": self.gate,
            "mu": self.mu,
            "sigma": self.sigma,
        }

    def fit_conditional(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        add_interactions: bool = False,
    ):
        """M1–M3: logistic gate + Gaussian expert conditioned on features."""
        X = self._prepare_features(df_train, features, add_interactions)
        y_exp = df_train["exposure_flag"].values.astype(int)

        # Gate: logistic regression
        if X.shape[1] > 0 and y_exp.sum() > 0 and (y_exp == 0).sum() > 0:
            self.gate_model = LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0
            )
            self.gate_model.fit(X, y_exp)
        else:
            self.gate = float(y_exp.mean()) if len(y_exp) > 0 else 0.0

        # Expert: Gaussian on events only
        event_mask = df_train["target_status"] == "event"
        y_events = df_train.loc[event_mask, "y_log_ttc"].dropna()

        if len(y_events) > 1:
            self.mu = float(y_events.mean())
            self.sigma = float(y_events.std())
        else:
            self.mu = float(math.log(T_MAX_S))
            self.sigma = 1.0

        self.config = {
            "type": "conditional",
            "n_features": X.shape[1],
            "features": features,
            "add_interactions": add_interactions,
            "mu": self.mu,
            "sigma": self.sigma,
        }

    def fit_gate_expert(
        self,
        df_train: pd.DataFrame,
        features: List[str],
    ):
        """M4: separate gate and expert."""
        X = self._prepare_features(df_train, features)
        y_exp = df_train["exposure_flag"].values.astype(int)

        # Gate
        if X.shape[1] > 0 and y_exp.sum() > 0 and (y_exp == 0).sum() > 0:
            self.gate_model = LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0
            )
            self.gate_model.fit(X, y_exp)

        # Expert: fit per-feature linear model on events
        event_mask = df_train["target_status"] == "event"
        df_events = df_train[event_mask]
        y_events = df_events["y_log_ttc"].dropna()

        if len(y_events) > 1:
            self.mu = float(y_events.mean())
            self.sigma = float(y_events.std())
        else:
            self.mu = float(math.log(T_MAX_S))
            self.sigma = 1.0

        self.config = {
            "type": "gate_expert",
            "n_features": X.shape[1],
            "features": features,
            "mu": self.mu,
            "sigma": self.sigma,
        }

    def predict(
        self,
        df: pd.DataFrame,
        features: List[str],
        add_interactions: bool = False,
    ) -> pd.DataFrame:
        """Predict gate probability, expert mu/sigma, and risk score."""
        X = self._prepare_features(df, features, add_interactions)

        # Gate probability
        if self.gate_model is not None and X.shape[1] > 0:
            p_e = self.gate_model.predict_proba(X)[:, 1]
        else:
            p_e = np.full(len(df), self.gate if self.gate else 0.5)

        # Expert predictions (homoscedastic for now)
        pred_mu = np.full(len(df), self.mu)
        pred_sigma = np.full(len(df), self.sigma)

        # Risk score: R_τ = p_e · F_θ(log(τ) | C, Z)
        # P(TTC ≤ τ) = p_e · Φ((log(τ) - μ) / σ) for a given τ
        # We compute at τ = T_MAX_S as a summary
        log_tau = math.log(T_MAX_S)
        z_score = (log_tau - pred_mu) / np.maximum(pred_sigma, 1e-12)
        f_tau = norm.cdf(z_score)
        risk_score = p_e * f_tau

        result = df[["scenario_id", "time_index"]].copy()
        result["pred_p_exposure"] = p_e
        result["pred_mu"] = pred_mu
        result["pred_sigma"] = pred_sigma
        result["pred_risk_score"] = risk_score
        return result

    def _prepare_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        add_interactions: bool = False,
    ) -> np.ndarray:
        """Prepare feature matrix, using z-scored columns where available."""
        cols = []
        for f in features:
            z_col = f"{f}_z"
            if z_col in df.columns:
                cols.append(z_col)
            elif f in df.columns:
                cols.append(f)

        if not cols:
            return np.zeros((len(df), 0))

        X = df[cols].fillna(0).values.astype(np.float64)

        if add_interactions:
            # Add C_ODD × Z_state interactions
            c_idx = [i for i, f in enumerate(features) if f in C_ODD_FEATURES]
            z_idx = [i for i, f in enumerate(features) if f in Z_STATE_FEATURES]
            interactions = []
            for ci in c_idx[:4]:   # limit to first 4 C features
                for zi in z_idx:
                    if ci < X.shape[1] and zi < X.shape[1]:
                        interactions.append(X[:, ci] * X[:, zi])
            if interactions:
                X = np.column_stack([X] + interactions)

        return X


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_all_models(
    output_root: str,
    max_scenarios: Optional[int] = None,
    womd_root: Optional[str] = None,
) -> Dict:
    """Train M0–M5 and save results."""
    model_dir = output_root
    os.makedirs(model_dir, exist_ok=True)

    # Load model table
    table_path = os.path.join(output_root, "model_table.parquet")
    if not os.path.isfile(table_path):
        print("[Build] Model table not found, building...")
        if womd_root is None:
            womd_root = os.environ.get("WOMD_ROOT", "/mnt/womd")
        build_model_table(
            womd_root=womd_root,
            output_root=output_root,
            splits=["training"],
            max_scenarios=max_scenarios,
        )

    print("\n[Load] Reading model table...")
    df = pd.read_parquet(table_path)
    print(f"  Total rows: {len(df)}, scenarios: {df['scenario_id'].nunique()}")

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "internal_val"]

    print(f"  Train: {len(df_train)} rows ({df_train['scenario_id'].nunique()} scenarios)")
    print(f"  Val:   {len(df_val)} rows ({df_val['scenario_id'].nunique()} scenarios)")

    results = {}

    for model_key, spec in MODEL_SPECS.items():
        print(f"\n{'='*40}")
        print(f"Training {model_key}: {spec['name']}")
        print(f"{'='*40}")

        if spec.get("placeholder"):
            print(f"  [SKIP] {model_key} is a placeholder (warp disabled)")
            results[model_key] = {
                "status": "placeholder",
                "description": spec["description"],
            }
            continue

        model = HurdleModel(model_key)
        features = spec["features"]

        if model_key == "M0":
            model.fit_unconditional(df_train)
        elif spec.get("separate_gate_expert"):
            model.fit_gate_expert(df_train, features)
        else:
            add_int = spec.get("interactions", False)
            model.fit_conditional(df_train, features, add_interactions=add_int)

        # Predict on train + val
        add_int = spec.get("interactions", False)
        pred_train = model.predict(df_train, features, add_interactions=add_int)
        pred_val = model.predict(df_val, features, add_interactions=add_int)

        # Compute metrics
        metrics_train = _compute_metrics(df_train, pred_train, model, "train")
        metrics_val = _compute_metrics(df_val, pred_val, model, "internal_val")

        print(f"  Train NLL: {metrics_train.get('censored_nll', 'N/A'):.4f}")
        print(f"  Val   NLL: {metrics_val.get('censored_nll', 'N/A'):.4f}")
        print(f"  Train Brier: {metrics_train.get('gate_brier', 'N/A'):.4f}")
        print(f"  Val   Brier: {metrics_val.get('gate_brier', 'N/A'):.4f}")

        results[model_key] = {
            "status": "trained",
            "config": model.config,
            "description": spec["description"],
            "metrics_train": metrics_train,
            "metrics_val": metrics_val,
        }

    # Save model configs
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[OK] Model config: {config_path}")

    # Save metrics CSV
    _save_metrics_csv(results, model_dir)

    # Save frame predictions for best model
    if df_val is not None and not df_val.empty:
        best_key = _find_best_model(results)
        if best_key and best_key != "M5":
            spec = MODEL_SPECS[best_key]
            model = HurdleModel(best_key)
            features = spec["features"]

            if best_key == "M0":
                model.fit_unconditional(df_train)
            elif spec.get("separate_gate_expert"):
                model.fit_gate_expert(df_train, features)
            else:
                add_int = spec.get("interactions", False)
                model.fit_conditional(df_train, features, add_interactions=add_int)

            pred_all = model.predict(
                df, features,
                add_interactions=spec.get("interactions", False),
            )
            pred_all["split"] = df["split"].values
            pred_all["y_log_ttc"] = df["y_log_ttc"].values
            pred_all["ttc_censored"] = df["ttc_censored"].values
            pred_all["exposure_flag"] = df["exposure_flag"].values
            pred_all["target_status"] = df["target_status"].values

            pred_path = os.path.join(model_dir, "frame_predictions.parquet")
            pred_all.to_parquet(pred_path, index=False)
            print(f"[OK] Frame predictions: {pred_path}")

    print(f"\n[DONE] All models trained.")
    return results


def _compute_metrics(
    df: pd.DataFrame,
    pred: pd.DataFrame,
    model: HurdleModel,
    split_name: str,
) -> Dict:
    """Compute evaluation metrics for a model on a split."""
    metrics = {"split": split_name}

    # Exposure gate Brier score
    if "pred_p_exposure" in pred.columns and "exposure_flag" in df.columns:
        p_e = pred["pred_p_exposure"].values
        y_e = df["exposure_flag"].values.astype(float)
        valid = np.isfinite(p_e) & np.isfinite(y_e)
        if valid.sum() > 0:
            metrics["gate_brier"] = float(
                np.mean((p_e[valid] - y_e[valid]) ** 2)
            )

    # Censored NLL (on exposed frames only)
    exposed = df["exposure_flag"] == 1
    if exposed.sum() > 0:
        y = df.loc[exposed, "y_log_ttc"].values
        delta = (df.loc[exposed, "target_status"] == "event").values.astype(float)
        mu = pred.loc[exposed.values, "pred_mu"].values if "pred_mu" in pred.columns else np.full(exposed.sum(), model.mu)
        sigma = pred.loc[exposed.values, "pred_sigma"].values if "pred_sigma" in pred.columns else np.full(exposed.sum(), model.sigma)

        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
        if valid.sum() > 0:
            metrics["censored_nll"] = censored_nll(
                y[valid], delta[valid], mu[valid], sigma[valid]
            )

    return metrics


def _save_metrics_csv(results: Dict, output_dir: str):
    """Save model metrics as CSV."""
    rows = []
    for model_key, res in results.items():
        if res.get("status") == "placeholder":
            rows.append({"model": model_key, "status": "placeholder"})
            continue
        for split_key in ["metrics_train", "metrics_val"]:
            m = res.get(split_key, {})
            row = {"model": model_key, **m}
            rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "model_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Model metrics: {path}")


def _find_best_model(results: Dict) -> Optional[str]:
    """Find model with lowest val censored NLL."""
    best_key = None
    best_nll = float("inf")
    for key, res in results.items():
        if res.get("status") == "placeholder":
            continue
        val_nll = res.get("metrics_val", {}).get("censored_nll", float("inf"))
        if val_nll < best_nll:
            best_nll = val_nll
            best_key = key
    return best_key


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2 Model Training")
    ap.add_argument("--womd-root",
                    default=os.environ.get("WOMD_ROOT", "/mnt/womd"))
    ap.add_argument("--output-root",
                    default=os.path.join(
                        os.environ.get("PHASE2_OUTPUT_ROOT", "/mnt/phase2_outputs"),
                        "model",
                    ))
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()

    train_all_models(
        output_root=args.output_root,
        max_scenarios=args.max_scenarios,
        womd_root=args.womd_root,
    )


if __name__ == "__main__":
    main()
