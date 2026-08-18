#!/usr/bin/env python3
"""
Static Map Feature Extraction
===============================
Extracts map_features from a Scenario proto into flat rows suitable
for the map_feature Parquet table.

Each MapFeature may contain one of:
  lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, driveway

Polyline features → one row per point.
Point features (stop_sign) → a single row.

sdc_paths has 0% coverage — route features are NOT fabricated.
"""

from typing import Dict, List

import numpy as np


def extract_map_features(scenario, scenario_id: str) -> List[Dict]:
    """Extract map features from a Scenario proto.

    Returns a list of dicts ready for DataFrame construction, each with:
      scenario_id, feature_id, feature_type, point_index, x, y, z
    """
    rows = []

    for mf in scenario.map_features:
        feature_id = mf.id
        feature_type, polyline = _get_feature_type_and_polyline(mf)

        if polyline is not None:
            for pt_idx, pt in enumerate(polyline):
                rows.append({
                    "scenario_id": scenario_id,
                    "feature_id": feature_id,
                    "feature_type": feature_type,
                    "point_index": pt_idx,
                    "x": pt.x,
                    "y": pt.y,
                    "z": pt.z,
                })
        elif feature_type == "stop_sign":
            # Stop sign has a position point, not a polyline
            ss = mf.stop_sign
            if ss.HasField("position"):
                rows.append({
                    "scenario_id": scenario_id,
                    "feature_id": feature_id,
                    "feature_type": feature_type,
                    "point_index": 0,
                    "x": ss.position.x,
                    "y": ss.position.y,
                    "z": ss.position.z,
                })
            else:
                rows.append({
                    "scenario_id": scenario_id,
                    "feature_id": feature_id,
                    "feature_type": feature_type,
                    "point_index": 0,
                    "x": np.nan,
                    "y": np.nan,
                    "z": np.nan,
                })

    return rows


def _get_feature_type_and_polyline(mf):
    """Determine which oneof is set and return (type_str, polyline_or_None)."""
    if mf.HasField("lane"):
        return "lane", mf.lane.polyline
    elif mf.HasField("road_line"):
        return "road_line", mf.road_line.polyline
    elif mf.HasField("road_edge"):
        return "road_edge", mf.road_edge.polyline
    elif mf.HasField("crosswalk"):
        return "crosswalk", mf.crosswalk.polygon
    elif mf.HasField("speed_bump"):
        return "speed_bump", mf.speed_bump.polygon
    elif mf.HasField("stop_sign"):
        return "stop_sign", None
    elif mf.HasField("driveway"):
        return "driveway", mf.driveway.polygon
    else:
        return "unknown", None


def get_map_coverage_stats(rows: List[Dict]) -> Dict:
    """Compute coverage stats for map features."""
    if not rows:
        return {
            "total_points": 0,
            "feature_types": {},
            "n_features": 0,
        }

    feature_ids = set()
    type_counts = {}
    for r in rows:
        feature_ids.add(r["feature_id"])
        ft = r["feature_type"]
        type_counts[ft] = type_counts.get(ft, 0) + 1

    return {
        "total_points": len(rows),
        "n_features": len(feature_ids),
        "feature_types": type_counts,
    }
