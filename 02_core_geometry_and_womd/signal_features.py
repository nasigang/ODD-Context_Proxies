#!/usr/bin/env python3
"""
Dynamic Signal State Extraction
=================================
Extracts dynamic_map_states from a Scenario proto into flat rows
for the dynamic_signal Parquet table.

Each timestep may contain multiple DynamicMapState entries, each with:
  lane_id, state (signal colour/arrow), stop_point.
"""

from typing import Dict, List

from phase2_womd.schema import SIGNAL_STATE_MAP


def extract_dynamic_signals(scenario, scenario_id: str) -> List[Dict]:
    """Extract traffic signal states across all timesteps.

    Returns a list of dicts, each with:
      scenario_id, time_index, lane_id, signal_state, stop_point_x, stop_point_y
    """
    rows = []

    for time_idx, dms in enumerate(scenario.dynamic_map_states):
        for lane_state in dms.lane_states:
            state_value = lane_state.state
            state_str = SIGNAL_STATE_MAP.get(state_value, f"UNKNOWN_{state_value}")

            stop_x = None
            stop_y = None
            if lane_state.HasField("stop_point"):
                stop_x = lane_state.stop_point.x
                stop_y = lane_state.stop_point.y

            rows.append({
                "scenario_id": scenario_id,
                "time_index": time_idx,
                "lane_id": lane_state.lane,
                "signal_state": state_str,
                "stop_point_x": stop_x,
                "stop_point_y": stop_y,
            })

    return rows


def get_signal_coverage_stats(rows: List[Dict], n_timestamps: int) -> Dict:
    """Compute signal coverage stats."""
    if not rows:
        return {
            "total_entries": 0,
            "timesteps_with_signals": 0,
            "unique_lanes": 0,
            "signal_coverage_pct": 0.0,
        }

    timesteps_with = len(set(r["time_index"] for r in rows))
    unique_lanes = len(set(r["lane_id"] for r in rows))

    return {
        "total_entries": len(rows),
        "timesteps_with_signals": timesteps_with,
        "unique_lanes": unique_lanes,
        "signal_coverage_pct": round(
            100.0 * timesteps_with / n_timestamps, 2
        ) if n_timestamps > 0 else 0.0,
    }
