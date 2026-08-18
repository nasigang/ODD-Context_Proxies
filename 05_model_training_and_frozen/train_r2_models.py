#!/usr/bin/env python3
"""
R2 Production Training — gate-attested, fail-closed.

Requires:
1. Gate attestation (from R2InputGate.create_attestation)
2. Attestation re-verified before any .fit()
3. No fillna(0), no membership regeneration
4. Optimizer failure blocks artifact save
5. All-censored → ModelIdentifiabilityError
"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2_womd.r2_models import CensoredLogNormalAFT
from phase2_womd.r2_censored_likelihood import NLL_LABELS
from phase2_womd.r2_input_gate import R2InputGate, GateAttestationError
from phase2_womd.r2_split import load_frozen_membership
from phase2_womd.r2_feature_engineering import FEATURE_REGISTRY, check_leakage


class ModelIdentifiabilityError(Exception):
    pass


class TrainingError(Exception):
    pass


# Feature contract (fixed for this task)
Z_STATE_FEATURES = [
    "ego_speed_mps", "ego_accel_mps2", "ego_yaw_rate_rps",
    "n_eligible_pairs", "min_pair_distance_m", "max_closing_speed_mps",
]
C_CONTEXT_FEATURES = ["traffic_n_valid_agents"]
ALL_FEATURES = Z_STATE_FEATURES + C_CONTEXT_FEATURES

# Model specifications
MODEL_SPECS = {
    "M0": {"features": []},                 # intercept only
    "M1": {"features": Z_STATE_FEATURES},   # state only
    "M2": {"features": C_CONTEXT_FEATURES}, # context only
    "M3": {"features": ALL_FEATURES},       # state + context
}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def train_r2_models(attestation_path, r1_dir, frame_path, membership_path,
                    output_dir, maxiter=500):
    """
    Production R2 training with gate attestation verification.

    1. Re-verify attestation
    2. Load frame targets and membership
    3. Build model table (explicit joins)
    4. Train M0-M3 on train split only
    5. Save artifacts only if optimizer succeeds
    """
    # ── Step 1: Re-verify attestation ──
    frame_hash = _sha256_file(frame_path) if os.path.exists(frame_path) else None
    membership_hash = _sha256_file(membership_path) if os.path.exists(membership_path) else None

    att = R2InputGate.verify_attestation(
        attestation_path, r1_dir, frame_hash, membership_hash)
    print(f"[GATE] Attestation verified: status={att['status']}")

    # ── Step 2: Load data ──
    import pandas as pd
    import pyarrow.parquet as pq

    df = pq.read_table(frame_path).to_pandas()

    # Load membership — REQUIRED, no fallback generation
    membership = load_frozen_membership(membership_path)
    scenario_to_split = {}
    for split_name, sids in membership.items():
        for sid in sids:
            if sid in scenario_to_split:
                raise TrainingError(f"Duplicate scenario in membership: {sid}")
            scenario_to_split[sid] = split_name

    # Assign splits to frames
    df["split"] = df["scenario_id"].map(scenario_to_split)
    unassigned = df["split"].isna()
    if unassigned.any():
        n_bad = unassigned.sum()
        bad_sids = df.loc[unassigned, "scenario_id"].unique()[:5]
        raise TrainingError(
            f"{n_bad} frames have unassigned scenarios ({len(bad_sids)} unique): {list(bad_sids)}")

    # ── Step 3: Primary risk set only ──
    primary = df[df["target_status"].isin({"future_contact_event", "right_censored"})].copy()
    if "overlap_now_flag" in primary.columns:
        if primary["overlap_now_flag"].any():
            raise TrainingError("Overlap rows in primary risk set")

    # Train split only
    train_df = primary[primary["split"] == "train"].copy()
    if len(train_df) == 0:
        raise TrainingError("No training rows in primary risk set")

    # Check for all-censored
    censored = train_df["target_status"] == "right_censored"
    if censored.all():
        raise ModelIdentifiabilityError(
            "All training observations are right-censored. "
            "Location parameter is not identifiable without any events.")

    # ── Step 4: Build features (NO fillna(0)) ──
    y_ttc = train_df["ttc_obb_swept_s"].values
    is_censored = censored.values
    scenario_ids = train_df["scenario_id"].values

    censor_time = None
    if "censor_time_s" in train_df.columns:
        censor_time = train_df["censor_time_s"].values

    # Check features exist and are finite for required rows
    missing_report = {}
    for feat in ALL_FEATURES:
        if feat not in train_df.columns:
            raise TrainingError(f"Required feature '{feat}' not in frame table")
        n_nan = train_df[feat].isna().sum()
        n_inf = np.isinf(train_df[feat].values.astype(float)).sum() if n_nan < len(train_df) else 0
        missing_report[feat] = {"n_nan": int(n_nan), "n_inf": int(n_inf)}

    # Apply per-feature missing policy: exclude rows with NaN in required features
    feature_mask = np.ones(len(train_df), dtype=bool)
    for feat in ALL_FEATURES:
        feat_vals = train_df[feat].values.astype(float)
        bad = ~np.isfinite(feat_vals)
        feature_mask &= ~bad

    n_excluded = (~feature_mask).sum()
    train_clean = train_df[feature_mask].copy()
    if len(train_clean) == 0:
        raise TrainingError(f"All {len(train_df)} rows excluded by missing policy")

    y_ttc_clean = train_clean["ttc_obb_swept_s"].values
    censored_clean = (train_clean["target_status"] == "right_censored").values
    scenario_clean = train_clean["scenario_id"].values
    censor_time_clean = train_clean["censor_time_s"].values if "censor_time_s" in train_clean.columns else None

    # Re-check all-censored after exclusion
    if censored_clean.all():
        raise ModelIdentifiabilityError(
            "All training observations are right-censored after missing policy exclusion.")

    # ── Step 5: Fit preprocessing (train only) ──
    from sklearn.preprocessing import StandardScaler
    X_full = train_clean[ALL_FEATURES].values.astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(X_full)
    X_scaled = scaler.transform(X_full)

    # Verify no NaN/Inf after scaling
    if np.any(~np.isfinite(X_scaled)):
        raise TrainingError("NaN/Inf in scaled features after preprocessing")

    # Save preprocessing
    os.makedirs(output_dir, exist_ok=True)
    preproc = {
        "feature_names": ALL_FEATURES,
        "feature_order": ALL_FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fit_split": "train",
        "n_train_samples": len(train_clean),
        "n_excluded": int(n_excluded),
        "missing_report": missing_report,
    }
    preproc_path = os.path.join(output_dir, "preprocessing.pkl")
    with open(preproc_path, "wb") as f:
        pickle.dump(preproc, f)

    # ── Step 6: Train models ──
    results = {}
    for model_name, spec in MODEL_SPECS.items():
        features = spec["features"]
        if not features:
            X_model = np.ones((len(X_scaled), 1))
        else:
            feature_idx = [ALL_FEATURES.index(f) for f in features]
            X_model = X_scaled[:, feature_idx]

        model = CensoredLogNormalAFT(name=model_name, feature_names=features)
        fit_info = model.fit(X_model, y_ttc_clean, censored_clean,
                             censor_time=censor_time_clean, maxiter=maxiter)

        if not fit_info.get("optimizer_success", False):
            print(f"[WARN] {model_name} optimizer did not converge — artifact NOT saved")
            results[model_name] = {"status": "OPTIMIZER_FAILED", "fit_info": fit_info}
            continue

        # Save model artifact
        model_path = os.path.join(output_dir, f"model_{model_name}.pkl")
        model.save(model_path)
        results[model_name] = {
            "status": "FITTED",
            "path": model_path,
            "sha256": _sha256_file(model_path),
            "fit_info": {k: v for k, v in fit_info.items()
                         if k not in ("hessian",)},
        }
        print(f"[TRAIN] {model_name}: NLL={fit_info.get('final_nll_recomputed', 'N/A'):.4f}, "
              f"sigma={model.sigma:.4f}")

    # ── Step 7: Save training report ──
    report = {
        "attestation_path": attestation_path,
        "frame_path": frame_path,
        "membership_path": membership_path,
        "n_primary": len(primary),
        "n_train": len(train_df),
        "n_excluded_missing": int(n_excluded),
        "n_train_clean": len(train_clean),
        "n_events": int((~censored_clean).sum()),
        "n_censored": int(censored_clean.sum()),
        "feature_order": ALL_FEATURES,
        "models": results,
        "nll_label": NLL_LABELS["ttc_scale"],
        "NONINFERENTIAL_PILOT_DO_NOT_REPORT": True,
    }
    with open(os.path.join(output_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report
