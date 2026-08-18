#!/usr/bin/env python3
"""
Censoring Logic Tests
======================
Verify that:
  - no_exposure ≠ right_censored (distinct statuses)
  - y_log_ttc = log(T_MAX_S) when right_censored
  - y_log_ttc = log(max(TTC, T_MIN_S)) when event
  - ttc_censored=True only for right_censored, NOT for no_exposure
  - All frames preserved after aggregation (no silent drops)
"""

import math

import numpy as np
import pandas as pd
import pytest

from phase2_womd.build_frame_targets import build_frame_targets
from phase2_womd.obb_ttc import T_MAX_S, T_MIN_S


# ---------------------------------------------------------------------------
# Fixtures: synthetic agent + pair DataFrames
# ---------------------------------------------------------------------------

def _make_agent_df(scenario_id, n_frames=5, sdc_track_id=1,
                   sdc_valid=True):
    """Create a simple agent_state DataFrame with SDC + one target."""
    rows = []
    for t in range(n_frames):
        # SDC
        rows.append({
            "scenario_id": scenario_id,
            "time_index": t,
            "timestamp_seconds": t * 0.1,
            "track_id": sdc_track_id,
            "object_type": "TYPE_VEHICLE",
            "valid": sdc_valid,
            "center_x": float(t * 10) if sdc_valid else np.nan,
            "center_y": 0.0 if sdc_valid else np.nan,
            "center_z": 0.0 if sdc_valid else np.nan,
            "velocity_x": 10.0 if sdc_valid else np.nan,
            "velocity_y": 0.0 if sdc_valid else np.nan,
            "heading": 0.0 if sdc_valid else np.nan,
            "length": 4.0 if sdc_valid else np.nan,
            "width": 2.0 if sdc_valid else np.nan,
            "height": 1.5 if sdc_valid else np.nan,
            "is_sdc": True,
            "is_object_of_interest": False,
        })
        # A second agent (non-SDC)
        rows.append({
            "scenario_id": scenario_id,
            "time_index": t,
            "timestamp_seconds": t * 0.1,
            "track_id": 2,
            "object_type": "TYPE_VEHICLE",
            "valid": True,
            "center_x": 100.0, "center_y": 0.0, "center_z": 0.0,
            "velocity_x": 0.0, "velocity_y": 0.0,
            "heading": 0.0,
            "length": 4.0, "width": 2.0, "height": 1.5,
            "is_sdc": False,
            "is_object_of_interest": False,
        })
    return pd.DataFrame(rows)


def _make_pair_df_event(scenario_id, time_index, ttc_val):
    """Create a pair_metrics row with an event."""
    return pd.DataFrame([{
        "scenario_id": scenario_id,
        "time_index": time_index,
        "timestamp_seconds": time_index * 0.1,
        "ego_track_id": 1,
        "target_track_id": 2,
        "target_object_type": "TYPE_VEHICLE",
        "pair_distance_m": 20.0,
        "derived_ttc_2d_s": ttc_val,
        "derived_dtc_m": 0.0,
        "derived_closing_speed_mps": 10.0,
        "derived_overlap_now": False,
        "derived_hit_future": True,
        "derived_pair_valid": True,
        "derived_ttc_status": "event",
    }])


def _make_pair_df_censored(scenario_id, time_index):
    """Create a pair_metrics row that is right-censored."""
    return pd.DataFrame([{
        "scenario_id": scenario_id,
        "time_index": time_index,
        "timestamp_seconds": time_index * 0.1,
        "ego_track_id": 1,
        "target_track_id": 2,
        "target_object_type": "TYPE_VEHICLE",
        "pair_distance_m": 50.0,
        "derived_ttc_2d_s": T_MAX_S,
        "derived_dtc_m": 5.0,
        "derived_closing_speed_mps": 0.5,
        "derived_overlap_now": False,
        "derived_hit_future": False,
        "derived_pair_valid": True,
        "derived_ttc_status": "right_censored",
    }])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStatusDistinction:
    """no_exposure and right_censored must be distinct statuses."""

    def test_no_exposure_vs_right_censored(self):
        """Empty pairs → no_exposure; distant pair → right_censored."""
        df_agent = _make_agent_df("s1", n_frames=2)

        # Frame 0: has a censored pair
        # Frame 1: no pairs at all
        df_pairs = _make_pair_df_censored("s1", 0)

        targets = build_frame_targets(df_agent, df_pairs, "s1", 1)
        assert len(targets) == 2

        t0 = targets[targets["time_index"] == 0].iloc[0]
        t1 = targets[targets["time_index"] == 1].iloc[0]

        assert t0["target_status"] == "right_censored"
        assert t1["target_status"] == "no_exposure"
        assert t0["target_status"] != t1["target_status"]

    def test_no_exposure_not_censored_flag(self):
        """no_exposure frames should have ttc_censored=False."""
        df_agent = _make_agent_df("s2", n_frames=1)
        df_pairs = pd.DataFrame()  # No pairs

        targets = build_frame_targets(df_agent, df_pairs, "s2", 1)
        t = targets.iloc[0]
        assert t["target_status"] == "no_exposure"
        assert t["ttc_censored"] == False  # NOT True


class TestYLogTTC:
    """y_log_ttc values follow the correct formulas."""

    def test_event_y_log_ttc(self):
        """Event: y_log_ttc = log(max(TTC, T_MIN_S))."""
        df_agent = _make_agent_df("s3", n_frames=1)
        ttc_val = 3.5
        df_pairs = _make_pair_df_event("s3", 0, ttc_val)

        targets = build_frame_targets(df_agent, df_pairs, "s3", 1)
        t = targets.iloc[0]
        assert t["target_status"] == "event"
        expected_y = math.log(max(ttc_val, T_MIN_S))
        assert abs(t["y_log_ttc"] - expected_y) < 1e-10

    def test_event_small_ttc_clipped(self):
        """TTC < T_MIN_S → y_log_ttc = log(T_MIN_S)."""
        df_agent = _make_agent_df("s4", n_frames=1)
        ttc_val = 0.0  # current overlap
        df_pairs = _make_pair_df_event("s4", 0, ttc_val)

        targets = build_frame_targets(df_agent, df_pairs, "s4", 1)
        t = targets.iloc[0]
        expected_y = math.log(T_MIN_S)
        assert abs(t["y_log_ttc"] - expected_y) < 1e-10

    def test_censored_y_log_ttc(self):
        """Right-censored: y_log_ttc = log(T_MAX_S)."""
        df_agent = _make_agent_df("s5", n_frames=1)
        df_pairs = _make_pair_df_censored("s5", 0)

        targets = build_frame_targets(df_agent, df_pairs, "s5", 1)
        t = targets.iloc[0]
        assert t["target_status"] == "right_censored"
        expected_y = math.log(T_MAX_S)
        assert abs(t["y_log_ttc"] - expected_y) < 1e-10
        assert t["ttc_censored"] == True

    def test_no_exposure_y_nan(self):
        """No exposure: y_log_ttc = NaN."""
        df_agent = _make_agent_df("s6", n_frames=1)
        df_pairs = pd.DataFrame()

        targets = build_frame_targets(df_agent, df_pairs, "s6", 1)
        t = targets.iloc[0]
        assert t["target_status"] == "no_exposure"
        assert pd.isna(t["y_log_ttc"])

    def test_invalid_ego_y_nan(self):
        """Invalid ego: y_log_ttc = NaN."""
        df_agent = _make_agent_df("s7", n_frames=1, sdc_valid=False)
        df_pairs = pd.DataFrame()

        targets = build_frame_targets(df_agent, df_pairs, "s7", 1)
        t = targets.iloc[0]
        assert t["target_status"] == "invalid_ego_state"
        assert pd.isna(t["y_log_ttc"])


class TestFramePreservation:
    """All frames must be preserved — never silently dropped."""

    def test_all_frames_preserved(self):
        """5 frames → 5 target rows, regardless of pair availability."""
        n_frames = 5
        df_agent = _make_agent_df("s8", n_frames=n_frames)

        # Only frame 2 has a pair
        df_pairs = _make_pair_df_event("s8", 2, 5.0)

        targets = build_frame_targets(df_agent, df_pairs, "s8", 1)
        assert len(targets) == n_frames

        # Frame 2 should be event, all others no_exposure
        for _, row in targets.iterrows():
            if row["time_index"] == 2:
                assert row["target_status"] == "event"
            else:
                assert row["target_status"] == "no_exposure"

    def test_mixed_statuses_preserved(self):
        """Mix of event, censored, no_exposure in one scenario."""
        n_frames = 4
        df_agent = _make_agent_df("s9", n_frames=n_frames)

        df_pairs = pd.concat([
            _make_pair_df_event("s9", 0, 2.0),
            _make_pair_df_censored("s9", 1),
            # frames 2, 3 have no pairs
        ], ignore_index=True)

        targets = build_frame_targets(df_agent, df_pairs, "s9", 1)
        assert len(targets) == n_frames

        statuses = dict(zip(targets["time_index"], targets["target_status"]))
        assert statuses[0] == "event"
        assert statuses[1] == "right_censored"
        assert statuses[2] == "no_exposure"
        assert statuses[3] == "no_exposure"


class TestTTCCensoredFlag:
    """ttc_censored flag correctness."""

    def test_event_not_censored(self):
        df_agent = _make_agent_df("s10", n_frames=1)
        df_pairs = _make_pair_df_event("s10", 0, 3.0)

        targets = build_frame_targets(df_agent, df_pairs, "s10", 1)
        assert targets.iloc[0]["ttc_censored"] == False

    def test_censored_flagged(self):
        df_agent = _make_agent_df("s11", n_frames=1)
        df_pairs = _make_pair_df_censored("s11", 0)

        targets = build_frame_targets(df_agent, df_pairs, "s11", 1)
        assert targets.iloc[0]["ttc_censored"] == True

    def test_no_exposure_not_censored(self):
        """no_exposure should NOT have ttc_censored=True."""
        df_agent = _make_agent_df("s12", n_frames=1)
        df_pairs = pd.DataFrame()

        targets = build_frame_targets(df_agent, df_pairs, "s12", 1)
        assert targets.iloc[0]["ttc_censored"] == False
        assert targets.iloc[0]["target_status"] == "no_exposure"
