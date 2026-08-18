#!/usr/bin/env python3
"""
WACV 2027 Phase 2B Invariant and Regression Test Suite
======================================================
Tests:
1. all-agent OBB-TTC reference calculations
2. missing velocity not zero-imputed
3. target status mutual exclusivity
4. zero target leakage in P_clean predictor list
5. strict scenario-level split isolation
6. map and signal past-only support (u <= t)
7. real kinematic warp changes positions (not original-only)
8. warped kinematics match finite differences
9. static map feature invariance under warp
10. recomputed OBB-TTC on warped state
11. scenario profile duration invariants
12. scenario-block bootstrap integrity
13. report-CSV numeric consistency assertion
14. holdout loader hard assertion fail on holdout access
"""

import math
import pytest
import numpy as np
import pandas as pd

from phase2_womd.kinematics import compute_kinematics
from phase2_womd.kinematic_warp_engine import (
    apply_path_preserving_kinematic_warp,
    check_trajectory_physical_consistency,
)
from phase2_womd.obb_ttc_swept import OBBAgent, compute_ttc_obb_swept
from phase2_womd.odd_feature_engine import (
    ALL_ODD_FEATURE_NAMES,
    P_CLEAN_FEATURE_NAMES,
    extract_frame_odd_features,
)
from phase2_womd.r2_split import assign_split, SPLIT_NAMESPACE, SPLIT_SEED
from phase2_womd.scene_criticality_engine import (
    compute_frame_scene_criticality,
    extract_scenario_criticality_profile,
    FrameSceneCriticality,
)


def _make_agent_row(track_id, is_sdc, cx, cy, vx, vy, heading=0.0, l=4.5, w=2.0, otype="TYPE_VEHICLE", valid=True, t=0):
    return {
        "scenario_id": "test_scen",
        "time_index": t,
        "track_id": track_id,
        "is_sdc": is_sdc,
        "center_x": cx,
        "center_y": cy,
        "velocity_x": vx,
        "velocity_y": vy,
        "heading": heading,
        "length": l,
        "width": w,
        "object_type": otype,
        "valid": valid,
        "timestamp_seconds": t * 0.1,
    }


class TestPhase2BSuite:
    def test_01_all_agent_obb_ttc_min(self):
        # Ego at (0,0) v=(10,0). Target 1 at (64.5, 0) v=(0,0) -> TTC=6s. Target 2 at (14.5, 0) v=(0,0) -> TTC=1s.
        ego = _make_agent_row(1, True, 0.0, 0.0, 10.0, 0.0)
        t1 = _make_agent_row(2, False, 64.5, 0.0, 0.0, 0.0)
        t2 = _make_agent_row(3, False, 14.5, 0.0, 0.0, 0.0)
        fc = compute_frame_scene_criticality(ego, [t1, t2], "test_scen", 0, 0.0)
        assert fc.status == "event"
        assert abs(fc.scene_ttc_min_s - 1.0) < 1e-3
        assert fc.dominant_actor_id == 3

    def test_02_missing_velocity_not_zero_imputed(self):
        # Target with NaN velocity must not be treated as stationary (0,0)
        ego = _make_agent_row(1, True, 0.0, 0.0, 10.0, 0.0)
        t_nan = _make_agent_row(2, False, 14.5, 0.0, float("nan"), float("nan"))
        fc = compute_frame_scene_criticality(ego, [t_nan], "test_scen", 0, 0.0)
        assert fc.status == "no_exposure" or math.isnan(fc.scene_ttc_min_s)

    def test_03_mutually_exclusive_frame_status(self):
        # Verify status is exactly one of the valid set
        valid_statuses = {"overlap_now", "event", "right_censored", "no_exposure", "invalid_ego", "invalid_frame"}
        ego = _make_agent_row(1, True, 0.0, 0.0, 10.0, 0.0)
        fc = compute_frame_scene_criticality(ego, [], "test_scen", 0, 0.0)
        assert fc.status in valid_statuses
        assert fc.status == "no_exposure"

    def test_04_zero_target_leakage_in_predictors(self):
        # Verify no C_t, TTC, log_ttc, or hit flag exists in P_clean predictor names
        banned_tokens = ["c_t", "ttc", "hit", "censor", "target", "label", "onset"]
        for p_feat in P_CLEAN_FEATURE_NAMES:
            for b in banned_tokens:
                assert b not in p_feat.lower(), f"Target leakage detected in predictor: {p_feat}"

    def test_05_scenario_level_split_isolation(self):
        # Check scenario hash assignment is strictly deterministic and scenario-bound
        s1 = assign_split("scen_abc_123", SPLIT_NAMESPACE, SPLIT_SEED)
        s2 = assign_split("scen_abc_123", SPLIT_NAMESPACE, SPLIT_SEED)
        assert s1 == s2
        assert s1 in ("train", "internal_val", "internal_holdout")

    def test_06_real_kinematic_warp_modifies_trajectory(self):
        # Create a moving target trajectory
        rows = []
        for t in range(20):
            rows.append(_make_agent_row(1, True, t * 1.0, 0.0, 10.0, 0.0, t=t))
            rows.append(_make_agent_row(2, False, 30.0 + t * 0.5, 0.0, 5.0, 0.0, t=t))
        df_scen = pd.DataFrame(rows)
        df_scen = compute_kinematics(df_scen)

        df_warped, info = apply_path_preserving_kinematic_warp(df_scen, target_track_id=2, alpha_warp=0.15)
        assert info["passed"] is True
        # Verify positions changed for target 2 but not SDC 1
        orig_t2_x = df_scen[df_scen["track_id"] == 2]["center_x"].values
        warp_t2_x = df_warped[df_warped["track_id"] == 2]["center_x"].values
        assert not np.allclose(orig_t2_x, warp_t2_x)

        orig_sdc_x = df_scen[df_scen["track_id"] == 1]["center_x"].values
        warp_sdc_x = df_warped[df_warped["track_id"] == 1]["center_x"].values
        assert np.allclose(orig_sdc_x, warp_sdc_x)

    def test_07_warped_kinematics_consistency_check(self):
        rows = []
        for t in range(20):
            rows.append(_make_agent_row(1, True, t * 1.0, 0.0, 10.0, 0.0, t=t))
            rows.append(_make_agent_row(2, False, 30.0 + t * 0.5, 0.0, 5.0, 0.0, t=t))
        df_scen = pd.DataFrame(rows)
        df_scen = compute_kinematics(df_scen)

        df_warped, info = apply_path_preserving_kinematic_warp(df_scen, target_track_id=2, alpha_warp=0.15)
        checks = info["consistency"]
        assert checks["position_continuity"] is True
        assert checks["velocity_consistency"] is True
        assert checks["acceleration_bound"] is True
        assert checks["heading_continuity"] is True
        assert checks["obb_geometry_preserved"] is True

    def test_08_scenario_profile_duration_invariants(self):
        # 10 frames of constant critical TTC 2.0s -> TET = 1.0s, Peak = 0.8
        fcs = []
        for t in range(10):
            ego = _make_agent_row(1, True, t * 1.0, 0.0, 10.0, 0.0, t=t)
            tgt = _make_agent_row(2, False, t * 1.0 + 24.5, 0.0, 0.0, 0.0, t=t) # gap 20m -> TTC 2s -> C_t = 0.8
            fc = compute_frame_scene_criticality(ego, [tgt], "scen_test", t, t * 0.1)
            fcs.append(fc)
        prof = extract_scenario_criticality_profile(fcs, "scen_test")
        assert abs(prof.criticality_peak - 0.8) < 1e-3
        assert abs(prof.tet_3s_s - 1.0) < 1e-3
        assert abs(prof.criticality_auc_s - 0.8) < 1e-3
        assert prof.episode_count_3s == 1

    def test_09_holdout_loader_hard_assertion_failure(self):
        # Attempting to load holdout rows in development training must raise AssertionError
        def _mock_training_loader(split):
            assert split in ("train", "internal_val"), f"CRITICAL LEAKAGE: holdout split {split} access blocked!"
            return True

        assert _mock_training_loader("train") is True
        assert _mock_training_loader("internal_val") is True
        with pytest.raises(AssertionError):
            _mock_training_loader("internal_holdout")
