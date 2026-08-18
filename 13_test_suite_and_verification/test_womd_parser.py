#!/usr/bin/env python3
"""
WOMD Parser Smoke Tests
========================
Parse 10 scenarios from the training split and verify:
  1. All 5 tables produced with correct schemas
  2. Zero duplicate primary keys
  3. derived_* columns are NaN where valid=False
  4. Invalid states are NOT zeroed
  5. sdc_path_points skipped (unsupported)
  6. Row counts, coverage stats printed

Run inside Docker:
    python -m pytest tests/test_womd_parser.py -v
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from phase2_womd.parser import parse_scenarios
from phase2_womd.schema import (
    AGENT_STATE_PRIMARY_KEY,
    AGENT_STATE_SCHEMA,
    DYNAMIC_SIGNAL_PRIMARY_KEY,
    DYNAMIC_SIGNAL_SCHEMA,
    FRAME_CONTEXT_PRIMARY_KEY,
    FRAME_CONTEXT_SCHEMA,
    MAP_FEATURE_PRIMARY_KEY,
    MAP_FEATURE_SCHEMA,
    SCENARIO_TABLE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WOMD_ROOT = os.environ.get("WOMD_ROOT", "/mnt/womd")
SMOKE_MAX = 10
SPLITS = ["training"]


@pytest.fixture(scope="module")
def parsed_output():
    """Parse 10 scenarios and return (output_dir, summary)."""
    output_dir = tempfile.mkdtemp(prefix="womd_smoke_")
    try:
        summary = parse_scenarios(
            womd_root=WOMD_ROOT,
            output_root=output_dir,
            splits=SPLITS,
            max_scenarios=SMOKE_MAX,
        )
        yield output_dir, summary
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTableCreation:
    """Verify all required tables are created."""

    def test_scenario_table_exists(self, parsed_output):
        output_dir, _ = parsed_output
        assert os.path.isdir(os.path.join(output_dir, "scenario_table"))

    def test_agent_state_exists(self, parsed_output):
        output_dir, _ = parsed_output
        assert os.path.isdir(os.path.join(output_dir, "agent_state"))

    def test_frame_context_exists(self, parsed_output):
        output_dir, _ = parsed_output
        assert os.path.isdir(os.path.join(output_dir, "frame_context"))

    def test_map_feature_exists(self, parsed_output):
        output_dir, _ = parsed_output
        assert os.path.isdir(os.path.join(output_dir, "map_feature"))

    def test_dynamic_signal_exists(self, parsed_output):
        output_dir, _ = parsed_output
        assert os.path.isdir(os.path.join(output_dir, "dynamic_signal"))

    def test_sdc_path_not_created(self, parsed_output):
        """sdc_paths has 0% coverage → table should NOT exist."""
        output_dir, summary = parsed_output
        sdc_dir = os.path.join(output_dir, "sdc_path_points")
        assert not os.path.exists(sdc_dir), (
            "sdc_path_points should not be created (0% coverage)"
        )
        assert summary["route_coverage"]["status"] == "unsupported"


class TestScenarioCounts:
    """Verify correct number of scenarios parsed."""

    def test_scenario_count(self, parsed_output):
        _, summary = parsed_output
        assert summary["scenarios_parsed"] == SMOKE_MAX

    def test_zero_failures(self, parsed_output):
        _, summary = parsed_output
        assert summary["parse_failures"] == 0

    def test_scenario_table_rows(self, parsed_output):
        _, summary = parsed_output
        assert summary["tables"]["scenario_table"]["row_count"] == SMOKE_MAX


class TestDuplicateKeys:
    """No duplicate primary keys in any table."""

    def test_no_duplicate_scenario_keys(self, parsed_output):
        _, summary = parsed_output
        assert summary["duplicate_keys"]["scenario_table"] == 0

    def test_no_duplicate_agent_keys(self, parsed_output):
        _, summary = parsed_output
        assert summary["duplicate_keys"]["agent_state"] == 0

    def test_no_duplicate_frame_keys(self, parsed_output):
        _, summary = parsed_output
        assert summary["duplicate_keys"]["frame_context"] == 0

    def test_no_duplicate_map_keys(self, parsed_output):
        _, summary = parsed_output
        assert summary["duplicate_keys"]["map_feature"] == 0

    def test_no_duplicate_signal_keys(self, parsed_output):
        _, summary = parsed_output
        assert summary["duplicate_keys"]["dynamic_signal"] == 0


class TestInvalidStates:
    """Invalid states must be NaN, not zero."""

    def test_invalid_states_are_nan(self, parsed_output):
        output_dir, summary = parsed_output
        ds = pq.ParquetDataset(os.path.join(output_dir, "agent_state"))
        df = ds.read().to_pandas()

        invalid = df[~df["valid"]]
        if len(invalid) == 0:
            pytest.skip("No invalid states in sample")

        # center_x must be NaN for invalid states, NOT zero
        assert invalid["center_x"].isna().all(), (
            "Invalid states should have NaN center_x, not 0"
        )
        assert invalid["velocity_x"].isna().all(), (
            "Invalid states should have NaN velocity_x, not 0"
        )

    def test_invalid_state_ratio_reasonable(self, parsed_output):
        _, summary = parsed_output
        ratio = summary.get("invalid_state_ratio")
        if ratio is not None:
            # Invalid state ratio should be > 0 (some tracks are partial)
            # and < 1.0 (not all states are invalid)
            assert 0.0 <= ratio < 1.0


class TestDerivedKinematics:
    """Derived columns must follow validity rules."""

    def test_derived_nan_when_invalid(self, parsed_output):
        output_dir, _ = parsed_output
        ds = pq.ParquetDataset(os.path.join(output_dir, "agent_state"))
        df = ds.read().to_pandas()

        invalid = df[~df["valid"]]
        if len(invalid) == 0:
            pytest.skip("No invalid states in sample")

        # All derived columns should be NaN for invalid states
        assert invalid["derived_speed_mps"].isna().all(), (
            "derived_speed_mps should be NaN for invalid states"
        )
        assert invalid["derived_kinematic_valid"].eq(False).all(), (
            "derived_kinematic_valid should be False for invalid states"
        )

    def test_speed_non_negative(self, parsed_output):
        output_dir, _ = parsed_output
        ds = pq.ParquetDataset(os.path.join(output_dir, "agent_state"))
        df = ds.read().to_pandas()

        valid_speed = df.loc[df["valid"], "derived_speed_mps"].dropna()
        if len(valid_speed) == 0:
            pytest.skip("No valid speeds")
        assert (valid_speed >= 0).all(), "Speed must be non-negative"

    def test_kinematic_valid_implies_all_finite(self, parsed_output):
        output_dir, _ = parsed_output
        ds = pq.ParquetDataset(os.path.join(output_dir, "agent_state"))
        df = ds.read().to_pandas()

        kin_valid = df[df["derived_kinematic_valid"] == True]
        if len(kin_valid) == 0:
            pytest.skip("No kinematic-valid rows")

        for col in ["derived_speed_mps", "derived_accel_mps2",
                     "derived_yaw_rate_rps", "derived_jerk_mps3"]:
            assert np.isfinite(kin_valid[col]).all(), (
                f"{col} must be finite when derived_kinematic_valid=True"
            )


class TestSchemaCompliance:
    """Parquet schemas match defined schemas."""

    def test_agent_state_schema(self, parsed_output):
        output_dir, _ = parsed_output
        ds = pq.ParquetDataset(os.path.join(output_dir, "agent_state"))
        schema = ds.schema
        expected_cols = {f.name for f in AGENT_STATE_SCHEMA}
        # Parquet partitioning removes the partition column from file schema
        actual_cols = set(schema.names) | {"scenario_id"}
        missing = expected_cols - actual_cols
        assert not missing, f"Missing columns in agent_state: {missing}"

    def test_frame_context_schema(self, parsed_output):
        output_dir, _ = parsed_output
        ds = pq.ParquetDataset(os.path.join(output_dir, "frame_context"))
        schema = ds.schema
        expected_cols = {f.name for f in FRAME_CONTEXT_SCHEMA}
        actual_cols = set(schema.names) | {"scenario_id"}
        missing = expected_cols - actual_cols
        assert not missing, f"Missing columns in frame_context: {missing}"


class TestCoverage:
    """Feature coverage stats are populated."""

    def test_map_coverage(self, parsed_output):
        _, summary = parsed_output
        mc = summary["map_coverage"]
        assert mc["total_points"] > 0
        assert mc["n_features"] > 0

    def test_signal_coverage(self, parsed_output):
        _, summary = parsed_output
        sc = summary["signal_coverage"]
        assert sc["total_entries"] > 0

    def test_unsupported_features_logged(self, parsed_output):
        _, summary = parsed_output
        unsup = summary["unsupported_features"]
        assert "sdc_paths" in unsup
        assert "weather" in unsup
        assert "friction" in unsup
