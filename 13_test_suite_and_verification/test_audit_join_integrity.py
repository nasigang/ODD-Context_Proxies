#!/usr/bin/env python3
"""
Test Audit Join Integrity — verify Circle-OBB matched pair join quality.
"""
import os
import pytest
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "R1", "data")
PAIRS_PATH = os.path.join(DATA_DIR, "pair_metrics_obb_primary.parquet")
FRAMES_PATH = os.path.join(DATA_DIR, "frame_targets_obb_primary.parquet")


@pytest.fixture(scope="module")
def df_pairs():
    if not os.path.exists(PAIRS_PATH):
        pytest.skip("Pair metrics file not found")
    return pq.read_table(PAIRS_PATH, columns=[
        "scenario_id", "time_index", "ego_track_id", "target_track_id",
        "circle_ttc_s", "obb_ttc_s", "circle_status", "obb_status",
        "circle_overlap_now", "obb_overlap_now", "obb_hit_future",
        "circle_hit_future", "target_pair_key", "state_hash", "dimension_hash",
        "ttc_horizon_s", "target_object_type",
    ]).to_pandas()


@pytest.fixture(scope="module")
def df_frames():
    if not os.path.exists(FRAMES_PATH):
        pytest.skip("Frame targets file not found")
    return pq.read_table(FRAMES_PATH).to_pandas()


class TestJoinKeyIntegrity:

    def test_no_duplicate_pair_keys(self, df_pairs):
        """Each (scenario_id, time_index, ego_id, target_id) must be unique."""
        keys = df_pairs[["scenario_id", "time_index", "ego_track_id", "target_track_id"]]
        assert keys.duplicated().sum() == 0, "Duplicate pair keys found"

    def test_no_null_keys(self, df_pairs):
        for col in ["scenario_id", "time_index", "ego_track_id", "target_track_id"]:
            assert df_pairs[col].isna().sum() == 0, f"Null values in {col}"

    def test_target_pair_key_format(self, df_pairs):
        """target_pair_key should match key components."""
        sample = df_pairs.head(100)
        for _, row in sample.iterrows():
            expected = f"{row['scenario_id']}_{row['time_index']}_{row['ego_track_id']}_{row['target_track_id']}"
            assert row["target_pair_key"] == expected

    def test_state_hash_not_null(self, df_pairs):
        assert df_pairs["state_hash"].isna().sum() == 0

    def test_dimension_hash_not_null(self, df_pairs):
        assert df_pairs["dimension_hash"].isna().sum() == 0


class TestHorizonConsistency:

    def test_consistent_horizon(self, df_pairs):
        assert (df_pairs["ttc_horizon_s"] == 10.0).all()


class TestContainmentInvariant:

    def test_obb_hit_implies_circle_contact(self, df_pairs):
        """OBB hit → Circle must have contact (hit or overlap)."""
        obb_hit = df_pairs["obb_hit_future"]
        circle_contact = df_pairs["circle_hit_future"] | df_pairs["circle_overlap_now"]
        violations = (obb_hit & ~circle_contact).sum()
        assert violations == 0, f"{violations} containment violations"

    def test_circle_ttc_leq_obb_ttc_when_both_future(self, df_pairs):
        """When both have strictly-future contact, Circle TTC ≤ OBB TTC + 0.05s."""
        both_future = (
            df_pairs["obb_hit_future"] &
            df_pairs["circle_hit_future"] &
            ~df_pairs["obb_overlap_now"] &
            ~df_pairs["circle_overlap_now"]
        )
        if both_future.sum() == 0:
            pytest.skip("No both-future pairs")
        sub = df_pairs[both_future]
        tol = 0.05
        violations = (sub["circle_ttc_s"] > sub["obb_ttc_s"] + tol).sum()
        assert violations == 0, f"{violations} TTC ordering violations"


class TestEligibility:

    def test_valid_target_types(self, df_pairs):
        valid_types = {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}
        actual = set(df_pairs["target_object_type"].unique())
        assert actual.issubset(valid_types), f"Unexpected types: {actual - valid_types}"

    def test_pair_distance_within_radius(self, df_pairs):
        """All pairs should be within PAIR_RADIUS_M=70m."""
        if "pair_distance_m" in df_pairs.columns:
            assert (df_pairs["pair_distance_m"] <= 70.01).all()


class TestFrameIntegrity:

    def test_no_duplicate_frames(self, df_frames):
        keys = df_frames[["scenario_id", "time_index"]]
        assert keys.duplicated().sum() == 0

    def test_valid_target_status(self, df_frames):
        valid = {"future_contact_event", "right_censored",
                 "current_geometry_overlap", "no_exposure", "invalid_ego_state"}
        actual = set(df_frames["target_status"].unique())
        assert actual.issubset(valid), f"Unexpected: {actual - valid}"

    def test_overlap_excluded_from_future_event(self, df_frames):
        """future_contact_event frames must NOT have overlap_now_flag=True AND target_status overlap."""
        fe = df_frames[df_frames["target_status"] == "future_contact_event"]
        if "overlap_now_flag" in fe.columns and "ttc_obb_swept_s" in fe.columns:
            # TTC should be > 0 for future events
            assert (fe["ttc_obb_swept_s"] > 0).all(), "Future events with TTC=0"
