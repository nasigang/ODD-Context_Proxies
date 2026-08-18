#!/usr/bin/env python3
"""
Unit Tests for Prospective Protocol & Invariants
=================================================
Tests all required synthetic scenarios:
1. Positive event (approaching pair transitions to TTC <= 3.0s in horizon)
2. Negative event (stable parallel/diverging pair with full 20-frame follow-up)
3. Overlap (index-time overlap -> excluded; future overlap -> positive event)
4. Incomplete follow-up (target track drops at frame t=15 -> incomplete)
5. Same-pair identity maintenance (pair identity locked to t_0 target)
6. Target-switch demonstration (frame-min triggers on intruding actor B, while target A is negative)
7. Exclusion reasons (out of radius, already critical at index time)
"""

import math
import numpy as np
import pandas as pd
import pytest

from phase2_womd.prospective_protocol import (
    LandmarkEligibility,
    ProspectiveLandmarkResult,
    evaluate_candidate_pair_at_index,
    extract_scenario_landmarks,
)
from phase2_womd.obb_ttc_swept import OBBAgent


def _build_synthetic_agent_df(
    n_frames: int = 91,
    ego_trajectory_fn=None,
    target_trajectories=None,
) -> pd.DataFrame:
    """
    Build a synthetic agent_state DataFrame for testing.
    
    Args:
        n_frames: number of frames (0..90)
        ego_trajectory_fn: function(t) -> dict of (cx, cy, vx, vy, heading, valid, length, width)
        target_trajectories: dict of track_id -> function(t) -> dict of (...)
    """
    rows = []
    
    for t in range(n_frames):
        # Ego (SDC)
        if ego_trajectory_fn is not None:
            ego_state = ego_trajectory_fn(t)
            if ego_state is not None:
                rows.append({
                    "scenario_id": "test_scenario",
                    "time_index": t,
                    "timestamp_seconds": t * 0.1,
                    "track_id": 0,
                    "object_type": "TYPE_VEHICLE",
                    "valid": ego_state.get("valid", True),
                    "center_x": ego_state["cx"],
                    "center_y": ego_state["cy"],
                    "velocity_x": ego_state["vx"],
                    "velocity_y": ego_state["vy"],
                    "heading": ego_state["heading"],
                    "length": ego_state.get("length", 4.5),
                    "width": ego_state.get("width", 2.0),
                    "is_sdc": True,
                })
        
        # Targets
        if target_trajectories:
            for tid, fn in target_trajectories.items():
                tgt_state = fn(t)
                if tgt_state is not None:
                    rows.append({
                        "scenario_id": "test_scenario",
                        "time_index": t,
                        "timestamp_seconds": t * 0.1,
                        "track_id": tid,
                        "object_type": tgt_state.get("type", "TYPE_VEHICLE"),
                        "valid": tgt_state.get("valid", True),
                        "center_x": tgt_state["cx"],
                        "center_y": tgt_state["cy"],
                        "velocity_x": tgt_state["vx"],
                        "velocity_y": tgt_state["vy"],
                        "heading": tgt_state["heading"],
                        "length": tgt_state.get("length", 4.5),
                        "width": tgt_state.get("width", 2.0),
                        "is_sdc": False,
                    })
                    
    return pd.DataFrame(rows)


class TestProspectiveSyntheticProtocol:
    
    def test_positive_event_transition(self):
        """
        Scenario: Ego starts at x=0 moving east at 10 m/s.
        Target starts at x=60 moving west at 5 m/s.
        At t=10 (index time): distance = 45m.
        Initial index TTC = 45 / 15 = 3.0s -> wait, let's place target at x=70:
        At t=10: ego at x=10, tgt at x=65 (distance=55m, TTC ~ 55/15 = 3.66s > 3.0s -> eligible!).
        By t=20: ego at x=20, tgt at x=60 (distance=40m, TTC ~ 40/15 = 2.66s <= 3.0s -> positive event!).
        """
        def ego_fn(t):
            return {"cx": 1.0 * t, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt_fn(t):
            # starts at x=70 at t=0, moves at -5 m/s (dx=-0.5 per frame)
            return {"cx": 70.0 - 0.5 * t, "cy": 0.0, "vx": -5.0, "vy": 0.0, "heading": math.pi}
            
        df = _build_synthetic_agent_df(ego_trajectory_fn=ego_fn, target_trajectories={1: tgt_fn})
        
        results = extract_scenario_landmarks(df, "test_pos", t_0=10, horizon_frames=20)
        assert len(results) == 1
        r = results[0]
        assert r.is_eligible is True
        assert r.label == "positive"
        assert r.event_time_index is not None
        assert 10 < r.event_time_index <= 30
        assert r.event_ttc_s <= 3.0
        assert r.event_type in ("TTC_LE_3S", "OVERLAP")

    def test_negative_event_complete_followup(self):
        """
        Scenario: Ego and Target traveling parallel in adjacent lanes at same speed.
        Distance ~ 10m constant, closing speed = 0, no collision.
        All 20 frames observed and valid -> label = negative.
        """
        def ego_fn(t):
            return {"cx": 1.0 * t, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt_fn(t):
            return {"cx": 1.0 * t, "cy": 10.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        df = _build_synthetic_agent_df(ego_trajectory_fn=ego_fn, target_trajectories={1: tgt_fn})
        
        results = extract_scenario_landmarks(df, "test_neg", t_0=10, horizon_frames=20)
        assert len(results) == 1
        r = results[0]
        assert r.is_eligible is True
        assert r.label == "negative"
        assert r.event_time_index is None

    def test_index_overlap_exclusion(self):
        """
        Scenario: Ego and Target already overlapping at index time t=10.
        Must be excluded from primary candidate landmarks (exclusion_reason = INDEX_OVERLAP).
        """
        def ego_fn(t):
            return {"cx": 0.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "heading": 0.0}
            
        def tgt_fn(t):
            return {"cx": 1.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "heading": 0.0}
            
        df = _build_synthetic_agent_df(ego_trajectory_fn=ego_fn, target_trajectories={1: tgt_fn})
        
        results = extract_scenario_landmarks(df, "test_overlap", t_0=10, horizon_frames=20)
        assert len(results) == 1
        r = results[0]
        assert r.is_eligible is False
        assert r.exclusion_reason == "INDEX_OVERLAP"
        assert r.label == "excluded"

    def test_future_overlap_positive_event(self):
        """
        Scenario: Target moves at 10 m/s ahead of ego (safe distance 45m at t=10 -> TTC=inf).
        At t=15, target suddenly brakes/stops at x=50m.
        At t=16, future TTC drops <= 3.0s -> triggers prospective positive event!
        """
        def ego_fn(t):
            # Moves 1m per frame (10 m/s)
            return {"cx": 1.0 * t, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt_fn(t):
            if t < 15:
                return {"cx": 1.0 * t + 35.0, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            else:
                return {"cx": 50.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "heading": 0.0}
            
        df = _build_synthetic_agent_df(ego_trajectory_fn=ego_fn, target_trajectories={1: tgt_fn})
        
        results = extract_scenario_landmarks(df, "test_fut_overlap", t_0=10, horizon_frames=20)
        assert len(results) == 1
        r = results[0]
        assert r.is_eligible is True
        assert r.label == "positive"
        assert r.event_type in ("OVERLAP", "TTC_LE_3S")
        assert r.event_time_index is not None
        assert 10 < r.event_time_index <= 30



    def test_incomplete_followup_track_lost(self):
        """
        Scenario: Target is eligible at t=10, but disappears at t=18 (e.g. occluded / left sensor range).
        No collision occurs before t=18.
        Must be classified as 'incomplete' (NOT negative).
        """
        def ego_fn(t):
            return {"cx": 1.0 * t, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt_fn(t):
            if t >= 18:
                return None  # Disappears from data
            return {"cx": 1.0 * t, "cy": 25.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        df = _build_synthetic_agent_df(ego_trajectory_fn=ego_fn, target_trajectories={1: tgt_fn})
        
        results = extract_scenario_landmarks(df, "test_incomplete", t_0=10, horizon_frames=20)
        assert len(results) == 1
        r = results[0]
        assert r.is_eligible is True
        assert r.label == "incomplete"
        assert r.incomplete_reason == "TARGET_LOST"
        assert r.incomplete_time_index == 18

    def test_same_pair_maintenance_with_crossing_traffic(self):
        """
        Scenario:
        SDC is tracking Target 1 (safe parallel car).
        Actor 2 crosses far behind or perpendicular without colliding with Target 1.
        Same-pair protocol for Target 1 must maintain Target 1's identity and label (negative),
        independent of other actors.
        """
        def ego_fn(t):
            return {"cx": 1.0 * t, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt1_fn(t):
            return {"cx": 1.0 * t, "cy": 15.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt2_fn(t):
            # Distant vehicle
            return {"cx": 1.0 * t, "cy": 50.0, "vx": 5.0, "vy": 0.0, "heading": 0.0}
            
        df = _build_synthetic_agent_df(
            ego_trajectory_fn=ego_fn,
            target_trajectories={1: tgt1_fn, 2: tgt2_fn}
        )
        
        results = extract_scenario_landmarks(df, "test_samepair", t_0=10, horizon_frames=20)
        assert len(results) == 2
        r1 = next(r for r in results if r.target_track_id == 1)
        r2 = next(r for r in results if r.target_track_id == 2)
        assert r1.label == "negative"
        assert r2.label == "negative"

    def test_target_switch_detection(self):
        """
        Demonstrates Target Switch distortion in frame-min protocol:
        - Target 1 (Track 1) is a safe vehicle cruising alongside Ego (negative under same-pair).
        - Intruding vehicle (Track 2) cuts aggressively in front of Ego at t=20 (critical event).
        Under Same-Pair Protocol:
          - Landmark (SDC, Track 1) -> label = 'negative'.
        Under Frame-Min Protocol:
          - At frame t=20, Frame-Min sees Track 2 and triggers 'positive'!
          - Frame-Min reports positive, but for Track 1 this is a DISAGREEMENT and TARGET SWITCH!
        """
        def ego_fn(t):
            return {"cx": 1.0 * t, "cy": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt1_safe(t):
            # Cruising safely 20m away laterally
            return {"cx": 1.0 * t, "cy": 20.0, "vx": 10.0, "vy": 0.0, "heading": 0.0}
            
        def tgt2_intruder(t):
            # Cuts across ego's lane directly ahead at t=20 (x=25, y=0)
            if t < 15:
                return {"cx": 50.0, "cy": 40.0, "vx": 0.0, "vy": -20.0, "heading": -math.pi / 2}
            else:
                # Arrives right in front of ego at t=20
                return {"cx": 22.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "heading": 0.0}
                
        df = _build_synthetic_agent_df(
            ego_trajectory_fn=ego_fn,
            target_trajectories={1: tgt1_safe, 2: tgt2_intruder}
        )
        
        from phase2_womd.prospective_protocol import extract_scenario_landmarks_and_contrast
        lms, contrast = extract_scenario_landmarks_and_contrast(df, "test_switch", t_0=10, horizon_frames=20)
        lm1 = next(r for r in lms if r.target_track_id == 1)
        
        assert lm1.label == "negative"  # Target 1 itself never became critical!
        assert contrast.open_set_frame_min_label == "positive"  # Open-set frame-min triggered positive due to intruder!
        assert contrast.anchor_vs_open_disagreement is True  # Disagreement at scenario level!
        assert contrast.open_target_switch_or_new_entrant is True  # Target switch confirmed!
        assert contrast.open_set_first_critical_track_id == 2  # Critical actor was 2, not 1!

    def test_out_of_radius_exclusion(self):
        """Target at distance > 70m at t=10 must be excluded with OUT_OF_RADIUS."""
        def ego_fn(t):
            return {"cx": 0.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "heading": 0.0}
            
        def tgt_fn(t):
            return {"cx": 100.0, "cy": 0.0, "vx": 0.0, "vy": 0.0, "heading": 0.0}
            
        df = _build_synthetic_agent_df(ego_trajectory_fn=ego_fn, target_trajectories={1: tgt_fn})
        results = extract_scenario_landmarks(df, "test_radius", t_0=10, horizon_frames=20)
        assert len(results) == 1
        assert results[0].is_eligible is False
        assert results[0].exclusion_reason == "OUT_OF_RADIUS"
