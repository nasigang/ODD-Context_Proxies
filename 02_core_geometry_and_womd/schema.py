#!/usr/bin/env python3
"""
WOMD Parquet Schema Definitions
================================
Column definitions and PyArrow schemas for all output tables.

Tables:
  scenario_table   — one row per scenario
  agent_state      — per-agent, per-timestep state + derived kinematics
  frame_context    — per-timestep aggregates (NO TTC/range/collision)
  map_feature      — static map polyline points
  dynamic_signal   — traffic signal states per timestep
  sdc_path_points  — SDC planned path (only if coverage > 0)

Naming conventions:
  - Raw fields keep proto names (center_x, velocity_x, …)
  - Derived variables use  derived_  or  inferred_  prefix
  - Compatibility aliases: segment_id=scenario_id, frame_label=time_index,
    obj_id=track_id
"""

import pyarrow as pa

# ---------------------------------------------------------------------------
# Compatibility aliases (used during writes as duplicate columns)
# ---------------------------------------------------------------------------
ALIAS_MAP = {
    "segment_id": "scenario_id",
    "frame_label": "time_index",
    "obj_id": "track_id",
}

# ---------------------------------------------------------------------------
# Object type enum → human-readable
# ---------------------------------------------------------------------------
OBJECT_TYPE_MAP = {
    0: "TYPE_UNSET",
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
    4: "TYPE_OTHER",
}

# ---------------------------------------------------------------------------
# Canonical WOMD Protocol Constants
# ---------------------------------------------------------------------------
# Derived from WOMD v1.3.1 91-timestep standard:
# 10-history frames (0..9) + current index 10 + 80-future frames (11..90)
CURRENT_TIME_INDEX: int = 10
N_TIMESTAMPS_STANDARD: int = 91
DT_STANDARD_SECONDS: float = 0.1
PROSPECTIVE_HORIZON_SECONDS: float = 2.0
PROSPECTIVE_HORIZON_FRAMES: int = 20
INDEX_TIME_RADIUS_M: float = 70.0
TTC_CRITICAL_THRESHOLD_S: float = 3.0


# ---------------------------------------------------------------------------
# Scenario Table  (1 row per scenario)
# ---------------------------------------------------------------------------
SCENARIO_TABLE_COLUMNS = [
    "scenario_id",
    "segment_id",          # alias
    "n_timestamps",
    "n_tracks",
    "sdc_track_index",
    "n_map_features",
    "n_dynamic_signal_steps",
    "n_objects_of_interest",
    "has_sdc_paths",       # bool
    "timestamp_start",
    "timestamp_end",
    "duration_seconds",
]

SCENARIO_TABLE_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("segment_id", pa.string()),
    ("n_timestamps", pa.int32()),
    ("n_tracks", pa.int32()),
    ("sdc_track_index", pa.int32()),
    ("n_map_features", pa.int32()),
    ("n_dynamic_signal_steps", pa.int32()),
    ("n_objects_of_interest", pa.int32()),
    ("has_sdc_paths", pa.bool_()),
    ("timestamp_start", pa.float64()),
    ("timestamp_end", pa.float64()),
    ("duration_seconds", pa.float64()),
])

# ---------------------------------------------------------------------------
# Agent State Table  (n_timestamps × n_tracks rows per scenario)
# ---------------------------------------------------------------------------
AGENT_STATE_COLUMNS_RAW = [
    "scenario_id",
    "segment_id",          # alias
    "time_index",
    "frame_label",         # alias
    "timestamp_seconds",
    "track_id",
    "obj_id",              # alias
    "object_type",
    "valid",
    "center_x",
    "center_y",
    "center_z",
    "velocity_x",
    "velocity_y",
    "heading",
    "length",
    "width",
    "height",
    "is_sdc",
    "is_object_of_interest",
]

AGENT_STATE_COLUMNS_DERIVED = [
    "derived_speed_mps",
    "derived_accel_mps2",
    "derived_yaw_rate_rps",
    "derived_jerk_mps3",
    "derived_kinematic_valid",
]

AGENT_STATE_SCHEMA = pa.schema([
    # --- keys ---
    ("scenario_id", pa.string()),
    ("segment_id", pa.string()),
    ("time_index", pa.int32()),
    ("frame_label", pa.int32()),
    ("timestamp_seconds", pa.float64()),
    ("track_id", pa.int32()),
    ("obj_id", pa.int32()),
    # --- raw state ---
    ("object_type", pa.string()),
    ("valid", pa.bool_()),
    ("center_x", pa.float64()),
    ("center_y", pa.float64()),
    ("center_z", pa.float64()),
    ("velocity_x", pa.float32()),
    ("velocity_y", pa.float32()),
    ("heading", pa.float32()),
    ("length", pa.float32()),
    ("width", pa.float32()),
    ("height", pa.float32()),
    ("is_sdc", pa.bool_()),
    ("is_object_of_interest", pa.bool_()),
    # --- derived kinematics ---
    ("derived_speed_mps", pa.float64()),
    ("derived_accel_mps2", pa.float64()),
    ("derived_yaw_rate_rps", pa.float64()),
    ("derived_jerk_mps3", pa.float64()),
    ("derived_kinematic_valid", pa.bool_()),
])

AGENT_STATE_PRIMARY_KEY = ["scenario_id", "time_index", "track_id"]

# ---------------------------------------------------------------------------
# Frame Context Table  (n_timestamps rows per scenario)
# NOTE: No TTC, range, closing_speed, or collision flags here.
# ---------------------------------------------------------------------------
FRAME_CONTEXT_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("segment_id", pa.string()),
    ("time_index", pa.int32()),
    ("frame_label", pa.int32()),
    ("timestamp_seconds", pa.float64()),
    ("n_valid_agents", pa.int32()),
    ("n_vehicles", pa.int32()),
    ("n_pedestrians", pa.int32()),
    ("n_cyclists", pa.int32()),
    ("n_other", pa.int32()),
])

FRAME_CONTEXT_PRIMARY_KEY = ["scenario_id", "time_index"]

# ---------------------------------------------------------------------------
# Map Feature Table  (variable rows per scenario)
# ---------------------------------------------------------------------------
MAP_FEATURE_TYPES = {
    "lane": "lane",
    "road_line": "road_line",
    "road_edge": "road_edge",
    "stop_sign": "stop_sign",
    "crosswalk": "crosswalk",
    "speed_bump": "speed_bump",
    "driveway": "driveway",
}

MAP_FEATURE_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("feature_id", pa.int64()),
    ("feature_type", pa.string()),
    ("point_index", pa.int32()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("z", pa.float64()),
])

MAP_FEATURE_PRIMARY_KEY = ["scenario_id", "feature_id", "point_index"]

# ---------------------------------------------------------------------------
# Dynamic Signal Table  (variable rows per scenario × timestep)
# ---------------------------------------------------------------------------
SIGNAL_STATE_MAP = {
    0: "LANE_STATE_UNKNOWN",
    1: "LANE_STATE_ARROW_STOP",
    2: "LANE_STATE_ARROW_CAUTION",
    3: "LANE_STATE_ARROW_GO",
    4: "LANE_STATE_STOP",
    5: "LANE_STATE_CAUTION",
    6: "LANE_STATE_GO",
    7: "LANE_STATE_FLASHING_STOP",
    8: "LANE_STATE_FLASHING_CAUTION",
}

DYNAMIC_SIGNAL_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("time_index", pa.int32()),
    ("lane_id", pa.int64()),
    ("signal_state", pa.string()),
    ("stop_point_x", pa.float64()),
    ("stop_point_y", pa.float64()),
])

DYNAMIC_SIGNAL_PRIMARY_KEY = ["scenario_id", "time_index", "lane_id"]

# ---------------------------------------------------------------------------
# SDC Path Points (only if sdc_paths field has coverage)
# ---------------------------------------------------------------------------
SDC_PATH_SCHEMA = pa.schema([
    ("scenario_id", pa.string()),
    ("point_index", pa.int32()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("z", pa.float64()),
])

SDC_PATH_PRIMARY_KEY = ["scenario_id", "point_index"]

# Field known to be unsupported per Prompt 0 audit
UNSUPPORTED_FIELDS = [
    "sdc_paths",
    "path_samples",
    "weather",
    "visibility",
    "friction",
]
