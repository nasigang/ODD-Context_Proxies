#!/usr/bin/env python3
"""
Unit and Invariant Tests for Scene-Level Criticality Profiling Engine
====================================================================
Tests:
- Single pair criticality
- Multiple pairs with deterministic minimum and tie-breaking
- Actor switch between consecutive frames (turnover without label distortion)
- Open-set new actor entry
- Exact TTC tie break (clearance -> dist -> track_id)
- Current overlap
- Right-censored exposure
- No exposure (0 targets)
- Invalid ego / frame
- Temporal duration consistency (10Hz vs 5Hz)
"""

import math
import pytest
import numpy as np

from phase2_womd.scene_criticality_engine import (
    compute_frame_scene_criticality,
    extract_scenario_criticality_profile,
    FrameSceneCriticality,
)


def _make_agent_dict(
    track_id: int,
    is_sdc: bool,
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    heading: float = 0.0,
    length: float = 4.5,
    width: float = 2.0,
    object_type: str = "TYPE_VEHICLE",
    valid: bool = True,
):
    return {
        "track_id": track_id,
        "is_sdc": is_sdc,
        "center_x": cx,
        "center_y": cy,
        "velocity_x": vx,
        "velocity_y": vy,
        "heading": heading,
        "length": length,
        "width": width,
        "object_type": object_type,
        "valid": valid,
    }


class TestSceneCriticalityEngine:
    def test_01_single_pair_positive(self):
        # Ego at (0, 0) moving +x at 10 m/s. Target at (24.5, 0) stationary.
        # Lengths = 4.5m -> gap = 20.0m -> TTC = 2.0s.
        ego = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgt = _make_agent_dict(2, False, 24.5, 0.0, 0.0, 0.0)
        
        fc = compute_frame_scene_criticality(ego, [tgt], "scen_1", 0, 0.0)
        assert fc.status == "event"
        assert fc.is_critical_3s is True
        assert abs(fc.scene_ttc_min_s - 2.0) < 1e-3
        assert abs(fc.severity_c_t - 0.8) < 1e-3
        assert fc.dominant_actor_id == 2
        assert fc.dominant_actor_type == "TYPE_VEHICLE"

    def test_02_multiple_pairs_min_ttc(self):
        # Ego moving +x at 10 m/s.
        # Tgt 1: at (64.5, 0) stationary -> gap 60m -> TTC 6.0s
        # Tgt 2: at (14.5, 0) stationary -> gap 10m -> TTC 1.0s
        # Tgt 3: at (34.5, 0) stationary -> gap 30m -> TTC 3.0s
        ego = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgt1 = _make_agent_dict(101, False, 64.5, 0.0, 0.0, 0.0)
        tgt2 = _make_agent_dict(102, False, 14.5, 0.0, 0.0, 0.0)
        tgt3 = _make_agent_dict(103, False, 34.5, 0.0, 0.0, 0.0)
        
        fc = compute_frame_scene_criticality(ego, [tgt1, tgt2, tgt3], "scen_1", 0, 0.0)
        assert fc.status == "event"
        assert abs(fc.scene_ttc_min_s - 1.0) < 1e-3
        assert fc.dominant_actor_id == 102
        assert fc.n_valid_targets_70m == 3

    def test_03_actor_switch_between_consecutive_frames(self):
        # Frame 0: Tgt A is closer and has smaller TTC
        # Frame 1: Tgt B closes fast and becomes smaller TTC
        ego_0 = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgtA_0 = _make_agent_dict(10, False, 24.5, 0.0, 0.0, 0.0) # gap 20m -> TTC 2.0s
        tgtB_0 = _make_agent_dict(20, False, 54.5, 0.0, 0.0, 0.0) # gap 50m -> TTC 5.0s
        
        fc0 = compute_frame_scene_criticality(ego_0, [tgtA_0, tgtB_0], "scen_switch", 0, 0.0)
        assert fc0.dominant_actor_id == 10
        assert abs(fc0.scene_ttc_min_s - 2.0) < 1e-3
        
        # Frame 1 (t=0.1s): Tgt B accelerates towards ego (-x at 40m/s)
        ego_1 = _make_agent_dict(1, True, 1.0, 0.0, 10.0, 0.0)
        tgtA_1 = _make_agent_dict(10, False, 24.5, 0.0, 0.0, 0.0) # gap 19m -> TTC 1.9s
        tgtB_1 = _make_agent_dict(20, False, 50.5, 0.0, -30.0, 0.0) # rel_v 40m/s -> gap 45m -> TTC 1.125s
        
        fc1 = compute_frame_scene_criticality(ego_1, [tgtA_1, tgtB_1], "scen_switch", 1, 0.1)
        assert fc1.dominant_actor_id == 20
        assert abs(fc1.scene_ttc_min_s - 1.125) < 1e-3
        
        # Profile extraction should record 1 turnover cleanly
        profile = extract_scenario_criticality_profile([fc0, fc1], "scen_switch", split="train")
        assert profile.dominant_actor_turnover_count == 1
        assert profile.exposed_frames == 2
        assert profile.criticality_peak > 0.8

    def test_04_open_set_new_actor_entry(self):
        # Frame 0: Tgt at 80m (out of 70m radius) -> no exposure
        ego_0 = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgt_0 = _make_agent_dict(10, False, 80.0, 0.0, 0.0, 0.0)
        fc0 = compute_frame_scene_criticality(ego_0, [tgt_0], "scen_entry", 0, 0.0)
        assert fc0.status == "no_exposure"
        assert math.isnan(fc0.scene_ttc_min_s)
        
        # Frame 1: Tgt moves to 60m -> enters 70m radius -> evaluated cleanly
        ego_1 = _make_agent_dict(1, True, 1.0, 0.0, 10.0, 0.0)
        tgt_1 = _make_agent_dict(10, False, 60.0, 0.0, 0.0, 0.0)
        fc1 = compute_frame_scene_criticality(ego_1, [tgt_1], "scen_entry", 1, 0.1)
        assert fc1.status == "right_censored" or fc1.status == "event"
        assert fc1.n_valid_targets_70m == 1

    def test_05_exact_ttc_tie_break(self):
        # Ego at (0, 0). Two identical targets at same distance (30m), same velocities.
        # Track ID 10 vs 20 -> tie-break must deterministically pick min track_id 10.
        ego = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgt1 = _make_agent_dict(20, False, 30.0, 0.0, 0.0, 0.0)
        tgt2 = _make_agent_dict(10, False, 30.0, 0.0, 0.0, 0.0)
        
        fc = compute_frame_scene_criticality(ego, [tgt1, tgt2], "scen_tie", 0, 0.0)
        assert fc.dominant_actor_id == 10

    def test_06_current_overlap(self):
        ego = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgt = _make_agent_dict(2, False, 1.0, 0.0, 0.0, 0.0) # center 1m with 4.5m length -> overlapping!
        
        fc = compute_frame_scene_criticality(ego, [tgt], "scen_ovl", 0, 0.0)
        assert fc.status == "event"
        assert fc.scene_ttc_min_s == 0.0
        assert fc.severity_c_t == 1.0

    def test_07_all_pairs_right_censored(self):
        # Target parallel to ego in adjacent lane, matching speed (no collision trajectory)
        ego = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        tgt = _make_agent_dict(2, False, 0.0, 4.0, 10.0, 0.0)
        
        fc = compute_frame_scene_criticality(ego, [tgt], "scen_cens", 0, 0.0)
        assert fc.status == "right_censored"
        assert fc.scene_ttc_min_s == 10.0
        assert fc.severity_c_t == 0.0
        assert fc.is_critical_3s is False

    def test_08_no_exposure_and_invalid_ego(self):
        # No targets
        ego = _make_agent_dict(1, True, 0.0, 0.0, 10.0, 0.0)
        fc_no = compute_frame_scene_criticality(ego, [], "scen_none", 0, 0.0)
        assert fc_no.status == "no_exposure"
        assert math.isnan(fc_no.severity_c_t)
        
        # Invalid ego
        fc_inv = compute_frame_scene_criticality(None, [], "scen_inv", 0, 0.0)
        assert fc_inv.status == "invalid_ego"

    def test_09_duration_consistency_10hz_vs_5hz(self):
        # Create a synthetic 2.0-second event (20 frames at 10Hz vs 10 frames at 5Hz)
        frames_10hz = []
        for i in range(20):
            ego = _make_agent_dict(1, True, i * 1.0, 0.0, 10.0, 0.0)
            tgt = _make_agent_dict(2, False, i * 1.0 + 24.5, 0.0, 0.0, 0.0) # gap 20m -> TTC 2.0s -> C_t = 0.8
            fc = compute_frame_scene_criticality(ego, [tgt], "scen_dur", i, i * 0.1)
            frames_10hz.append(fc)
            
        prof_10hz = extract_scenario_criticality_profile(frames_10hz, "scen_dur")
        
        # Subsample to 5Hz (step = 2)
        frames_5hz = []
        for i in range(0, 20, 2):
            ego = _make_agent_dict(1, True, i * 1.0, 0.0, 10.0, 0.0)
            tgt = _make_agent_dict(2, False, i * 1.0 + 24.5, 0.0, 0.0, 0.0)
            fc = compute_frame_scene_criticality(ego, [tgt], "scen_dur", i, i * 0.1)
            frames_5hz.append(fc)
            
        prof_5hz = extract_scenario_criticality_profile(frames_5hz, "scen_dur")
        
        # Check that total TET and AUC durations are consistent between 10Hz and 5Hz
        assert abs(prof_10hz.tet_3s_s - prof_5hz.tet_3s_s) < 0.25
        assert abs(prof_10hz.criticality_auc_s - prof_5hz.criticality_auc_s) < 0.25
        assert prof_10hz.episode_count_3s == 1
        assert prof_5hz.episode_count_3s == 1
