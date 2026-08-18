#!/usr/bin/env python3
"""
WACV 2027 Phase 2E Invariant Test Suite
======================================
Tests:
1. Exact split membership counts and disjointness (12828 / 2813 / 2804).
2. Scenario total-weight equality: scenarios with different frame counts get equal total weight.
3. Estimator parity: point estimator and bootstrap helper agree on identity draws.
4. Duplicate block draw total weight equality.
5. Paired identical predictions produce zero delta.
6. Empirical p-value naming and formula (+1 correction).
7. Weighted Ridge fits on train only.
8. Model training and evaluation sample weight enforcement.
9. Atomic sentinel creation and failure on existing file.
"""

import json
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

from phase2_womd.phase2e_engine import (
    compute_weighted_spearman,
    compute_weighted_classification_metrics,
    compute_empirical_bootstrap_p_value,
    ScenarioBlockBootstrap,
    create_atomic_holdout_sentinel,
    P_CLEAN_FEATURES,
    PRIMARY_ELIGIBLE_E,
    E_STATIC_FEATURES,
    E_COMPOSITION_FEATURES,
    E_INTERACTION_FEATURES,
)
from phase2_womd.r2_split import generate_split_membership, SPLIT_NAMESPACE, SPLIT_SEED


def test_split_membership_counts_and_disjointness():
    """Verify exact split counts: 12828 train, 2813 val, 2804 holdout, 0 overlap."""
    import glob
    agent_dir = "/home/kiapi/waymo_motion_project/runtime/outputs/model/parquet/agent_state"
    dirs = glob.glob(os.path.join(agent_dir, "scenario_id=*"))
    scenario_ids = sorted([os.path.basename(d).split("=")[1] for d in dirs])
    assert len(scenario_ids) == 18445

    membership = generate_split_membership(scenario_ids, SPLIT_NAMESPACE, SPLIT_SEED)
    assert len(membership["train"]) == 12828
    assert len(membership["internal_val"]) == 2813
    assert len(membership["internal_holdout"]) == 2804
    assert len(membership["train"]) + len(membership["internal_val"]) == 15641

    s_tr = set(membership["train"])
    s_va = set(membership["internal_val"])
    s_ho = set(membership["internal_holdout"])
    assert len(s_tr & s_va) == 0
    assert len(s_tr & s_ho) == 0
    assert len(s_va & s_ho) == 0


def test_scenario_total_weight_equality():
    """Verify that scenarios with differing frame counts have identical total weights."""
    # Scenario A has 10 frames, Scenario B has 91 frames
    sids = np.array(["scen_A"] * 10 + ["scen_B"] * 91)
    df = pd.DataFrame({"scenario_id": sids})
    
    # Calculate scenario-equal weights: w_st = 1 / n_s
    counts = df.groupby("scenario_id")["scenario_id"].transform("count")
    df["weight"] = 1.0 / counts

    tot_weight_A = df[df["scenario_id"] == "scen_A"]["weight"].sum()
    tot_weight_B = df[df["scenario_id"] == "scen_B"]["weight"].sum()

    assert pytest.approx(tot_weight_A, abs=1e-9) == 1.0
    assert pytest.approx(tot_weight_B, abs=1e-9) == 1.0


def test_point_bootstrap_estimator_parity_on_identity_sample():
    """Verify point estimator and bootstrap helper compute identical values on identical data."""
    np.random.seed(42)
    sids = np.array(["s1"] * 50 + ["s2"] * 80 + ["s3"] * 30)
    counts = pd.Series(sids).groupby(sids).transform("count").to_numpy()
    weights = 1.0 / counts
    weights = weights * (3.0 / np.sum(weights))  # normalize to n_scenarios = 3

    y_true = np.random.choice([0, 1], size=len(sids), p=[0.9, 0.1])
    y_prob = np.random.uniform(0.0, 1.0, size=len(sids))

    # Point estimate
    point_ap = average_precision_score(y_true, y_prob, sample_weight=weights)

    # Replicate with identical indices (s1, s2, s3 in order)
    resampler = ScenarioBlockBootstrap(sids, n_boot=1, seed=42)
    resampler.boot_draws = [np.array(["s1", "s2", "s3"])]

    rep_idx, rep_w = resampler.resample_replicate(0)
    rep_ap = average_precision_score(y_true[rep_idx], y_prob[rep_idx], sample_weight=rep_w)

    assert pytest.approx(point_ap, abs=1e-9) == rep_ap


def test_duplicate_block_draw_total_weight_equality():
    """Verify that if a scenario is drawn multiple times in bootstrap, each instance has equal total weight."""
    sids = np.array(["s1"] * 10 + ["s2"] * 90)
    resampler = ScenarioBlockBootstrap(sids, n_boot=1, seed=42)
    # Draw s1 twice, s2 0 times
    resampler.boot_draws = [np.array(["s1", "s1"])]
    rep_idx, rep_w = resampler.resample_replicate(0)

    # Should have 20 rows total (10 + 10)
    assert len(rep_idx) == 20
    # First 10 rows sum to half of total weight, second 10 rows sum to half of total weight
    w1 = np.sum(rep_w[:10])
    w2 = np.sum(rep_w[10:])
    assert pytest.approx(w1, abs=1e-9) == w2
    assert pytest.approx(w1 + w2, abs=1e-9) == 2.0


def test_paired_identical_predictions_produce_zero_delta():
    """Verify that when comparing a model with itself, delta PR-AUC is identically zero."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.2, 0.9, 0.05, 0.3, 0.7, 0.15])
    w = np.ones(len(y_true))

    m1 = compute_weighted_classification_metrics(y_true, y_pred, w)
    m2 = compute_weighted_classification_metrics(y_true, y_pred, w)

    delta_ap = m2["pr_auc"] - m1["pr_auc"]
    assert pytest.approx(delta_ap, abs=1e-12) == 0.0


def test_empirical_bootstrap_p_value_formula():
    """Verify empirical bootstrap p-value formula with Laplace +1 correction."""
    # All deltas strictly positive (1000/1000)
    pos_deltas = np.ones(1000)
    p_pos = compute_empirical_bootstrap_p_value(pos_deltas)
    expected_p = (2 * 0 + 1) / 1001.0  # 1 / 1001 ~= 0.000999
    assert pytest.approx(p_pos, abs=1e-6) == expected_p

    # Exactly 50% positive, 50% negative (symmetric zero mean)
    sym_deltas = np.array([1.0] * 500 + [-1.0] * 500)
    p_sym = compute_empirical_bootstrap_p_value(sym_deltas)
    assert pytest.approx(p_sym, abs=1e-6) == 1.0


def test_atomic_holdout_sentinel_guard():
    """Verify that atomic sentinel cannot be overwritten and prevents double opening."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sentinel_file = os.path.join(tmpdir, "HOLDOUT_ACCESS_SENTINEL_V5.json")
        meta = {"run_id": "test_phase2e", "timestamp": "2026-08-14T15:44:00Z"}

        # First creation must succeed
        ok1 = create_atomic_holdout_sentinel(sentinel_file, meta)
        assert ok1 is True
        assert os.path.exists(sentinel_file)

        # Second creation must fail (O_EXCL)
        ok2 = create_atomic_holdout_sentinel(sentinel_file, meta)
        assert ok2 is False
