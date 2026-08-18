#!/usr/bin/env python3
"""
Feature Engineering Engine for Same-Pair Prospective Criticality
================================================================
Extracts Physics Core and Context Core features from index-time (t_0=10)
and historical (t <= 10) observations ONLY.

Zero future information leakage guarantee:
- No future positions, velocities, or headings
- No future TTC or frame-min flags
- No event outcomes or target-switch flags
- Focal ego and focal target excluded from third-party context metrics
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from phase2_womd.schema import CURRENT_TIME_INDEX, INDEX_TIME_RADIUS_M
from phase2_womd.obb_ttc_swept import OBBAgent, compute_ttc_obb_swept


PHYSICS_FEATURE_NAMES = [
    "capped_index_ttc_s",
    "is_ttc_finite",
    "is_censored",
    "index_distance_m",
    "radial_closing_speed_mps",
    "ego_longitudinal_pos_m",
    "ego_lateral_pos_m",
    "ego_longitudinal_vel_mps",
    "ego_lateral_vel_mps",
    "ego_speed_mps",
    "target_speed_mps",
    "rel_heading_sin",
    "rel_heading_cos",
    "ego_accel_mps2",
    "ego_accel_valid",
    "target_accel_mps2",
    "target_accel_valid",
    "ego_length_m",
    "ego_width_m",
    "target_length_m",
    "target_width_m",
]

CONTEXT_FEATURE_NAMES = [
    "n_actors_30m",
    "n_actors_50m",
    "n_actors_70m",
    "n_vehicles_70m",
    "n_pedestrians_70m",
    "n_cyclists_70m",
    "nearest_third_party_dist_m",
    "mean_third_party_speed_mps",
    "std_third_party_speed_mps",
    "circular_heading_dispersion",
    "max_positive_closing_pressure",
    "agg_positive_closing_pressure",
    "context_empty",
]

ALL_FEATURE_NAMES = PHYSICS_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


def extract_features_for_landmark(
    ego_agent: OBBAgent,
    target_agent: OBBAgent,
    ego_row: Dict[str, Any],
    target_row: Dict[str, Any],
    third_party_agents: List[Tuple[OBBAgent, Dict[str, Any]]],
    index_ttc_s: float,
    index_closing_speed_mps: float,
) -> Dict[str, float]:
    """
    Extract physics and context feature vector for an eligible landmark pair.
    
    Args:
        ego_agent: SDC OBBAgent at t_0
        target_agent: Target OBBAgent at t_0
        ego_row: Raw/derived dictionary for SDC at t_0
        target_row: Raw/derived dictionary for Target at t_0
        third_party_agents: List of (OBBAgent, raw_dict) for all other valid actors at t_0
        index_ttc_s: Precomputed OBB TTC at t_0
        index_closing_speed_mps: Precomputed radial closing speed at t_0
    """
    features: Dict[str, float] = {}
    
    # ---------------------------------------------------------------------------
    # 1. Physics Core Features
    # ---------------------------------------------------------------------------
    capped_ttc = min(10.0, max(0.0, index_ttc_s)) if not math.isnan(index_ttc_s) else 10.0
    features["capped_index_ttc_s"] = float(capped_ttc)
    features["is_ttc_finite"] = 1.0 if (not math.isnan(index_ttc_s) and index_ttc_s < 10.0) else 0.0
    features["is_censored"] = 1.0 if (math.isnan(index_ttc_s) or index_ttc_s >= 10.0) else 0.0
    
    dx = target_agent.cx - ego_agent.cx
    dy = target_agent.cy - ego_agent.cy
    dist = math.sqrt(dx * dx + dy * dy)
    features["index_distance_m"] = float(dist)
    features["radial_closing_speed_mps"] = float(index_closing_speed_mps if not math.isnan(index_closing_speed_mps) else 0.0)
    
    # Transform relative coordinates to ego-heading frame
    # ego_heading: 0 along +x, pi/2 along +y
    cos_h = math.cos(ego_agent.heading)
    sin_h = math.sin(ego_agent.heading)
    
    # Longitudinal (forward along ego heading), Lateral (left perpendicular)
    features["ego_longitudinal_pos_m"] = float(dx * cos_h + dy * sin_h)
    features["ego_lateral_pos_m"] = float(-dx * sin_h + dy * cos_h)
    
    dvx = target_agent.vx - ego_agent.vx
    dvy = target_agent.vy - ego_agent.vy
    features["ego_longitudinal_vel_mps"] = float(dvx * cos_h + dvy * sin_h)
    features["ego_lateral_vel_mps"] = float(-dvx * sin_h + dvy * cos_h)
    
    features["ego_speed_mps"] = float(math.sqrt(ego_agent.vx ** 2 + ego_agent.vy ** 2))
    features["target_speed_mps"] = float(math.sqrt(target_agent.vx ** 2 + target_agent.vy ** 2))
    
    delta_heading = target_agent.heading - ego_agent.heading
    features["rel_heading_sin"] = float(math.sin(delta_heading))
    features["rel_heading_cos"] = float(math.cos(delta_heading))
    
    # Acceleration from history
    ego_acc = ego_row.get("derived_accel_mps2")
    if ego_acc is not None and not math.isnan(float(ego_acc)):
        features["ego_accel_mps2"] = float(ego_acc)
        features["ego_accel_valid"] = 1.0
    else:
        features["ego_accel_mps2"] = 0.0
        features["ego_accel_valid"] = 0.0
        
    tgt_acc = target_row.get("derived_accel_mps2")
    if tgt_acc is not None and not math.isnan(float(tgt_acc)):
        features["target_accel_mps2"] = float(tgt_acc)
        features["target_accel_valid"] = 1.0
    else:
        features["target_accel_mps2"] = 0.0
        features["target_accel_valid"] = 0.0
        
    features["ego_length_m"] = float(ego_agent.length)
    features["ego_width_m"] = float(ego_agent.width)
    features["target_length_m"] = float(target_agent.length)
    features["target_width_m"] = float(target_agent.width)
    
    # ---------------------------------------------------------------------------
    # 2. Context Core Features (Third-Party Actors Only)
    # ---------------------------------------------------------------------------
    n_actors = len(third_party_agents)
    if n_actors == 0:
        features["n_actors_30m"] = 0.0
        features["n_actors_50m"] = 0.0
        features["n_actors_70m"] = 0.0
        features["n_vehicles_70m"] = 0.0
        features["n_pedestrians_70m"] = 0.0
        features["n_cyclists_70m"] = 0.0
        features["nearest_third_party_dist_m"] = 70.0
        features["mean_third_party_speed_mps"] = 0.0
        features["std_third_party_speed_mps"] = 0.0
        features["circular_heading_dispersion"] = 0.0
        features["max_positive_closing_pressure"] = 0.0
        features["agg_positive_closing_pressure"] = 0.0
        features["context_empty"] = 1.0
        return features
        
    c30, c50, c70 = 0, 0, 0
    n_veh, n_ped, n_cyc = 0, 0, 0
    dists = []
    speeds = []
    cos_headings = []
    sin_headings = []
    closing_pressures = []
    
    for agent_3p, row_3p in third_party_agents:
        dx3 = agent_3p.cx - ego_agent.cx
        dy3 = agent_3p.cy - ego_agent.cy
        d3 = math.sqrt(dx3 * dx3 + dy3 * dy3)
        if d3 > 70.0:
            continue
            
        c70 += 1
        dists.append(d3)
        if d3 <= 50.0:
            c50 += 1
        if d3 <= 30.0:
            c30 += 1
            
        otype = row_3p.get("object_type", "TYPE_VEHICLE")
        if otype == "TYPE_VEHICLE":
            n_veh += 1
        elif otype == "TYPE_PEDESTRIAN":
            n_ped += 1
        elif otype == "TYPE_CYCLIST":
            n_cyc += 1
            
        spd = math.sqrt(agent_3p.vx ** 2 + agent_3p.vy ** 2)
        speeds.append(spd)
        cos_headings.append(math.cos(agent_3p.heading))
        sin_headings.append(math.sin(agent_3p.heading))
        
        # Closing speed from 3rd party to ego
        if d3 > 1e-6:
            u3x, u3y = dx3 / d3, dy3 / d3
            rel_vx3 = agent_3p.vx - ego_agent.vx
            rel_vy3 = agent_3p.vy - ego_agent.vy
            cl_spd = -(rel_vx3 * u3x + rel_vy3 * u3y)
            pos_cl = max(0.0, cl_spd)
            closing_pressures.append((pos_cl, d3))
            
    features["n_actors_30m"] = float(c30)
    features["n_actors_50m"] = float(c50)
    features["n_actors_70m"] = float(c70)
    features["n_vehicles_70m"] = float(n_veh)
    features["n_pedestrians_70m"] = float(n_ped)
    features["n_cyclists_70m"] = float(n_cyc)
    
    if dists:
        features["nearest_third_party_dist_m"] = float(min(dists))
        features["mean_third_party_speed_mps"] = float(np.mean(speeds))
        features["std_third_party_speed_mps"] = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        
        # Circular dispersion: 1 - R, R = sqrt(mean(cos)^2 + mean(sin)^2)
        mean_cos = np.mean(cos_headings)
        mean_sin = np.mean(sin_headings)
        r_length = math.sqrt(mean_cos ** 2 + mean_sin ** 2)
        features["circular_heading_dispersion"] = float(max(0.0, min(1.0, 1.0 - r_length)))
        
        max_press = max([p[0] for p in closing_pressures]) if closing_pressures else 0.0
        agg_press = sum([p[0] / max(1.0, p[1]) for p in closing_pressures]) if closing_pressures else 0.0
        features["max_positive_closing_pressure"] = float(max_press)
        features["agg_positive_closing_pressure"] = float(agg_press)
        features["context_empty"] = 0.0
    else:
        features["nearest_third_party_dist_m"] = 70.0
        features["mean_third_party_speed_mps"] = 0.0
        features["std_third_party_speed_mps"] = 0.0
        features["circular_heading_dispersion"] = 0.0
        features["max_positive_closing_pressure"] = 0.0
        features["agg_positive_closing_pressure"] = 0.0
        features["context_empty"] = 1.0
        
    return features


def generate_feature_dictionary_df() -> pd.DataFrame:
    """Generate canonical feature dictionary metadata."""
    dict_records = [
        # Physics Core
        {"feature_name": "capped_index_ttc_s", "role": "physics", "formula": "min(10.0, max(0.0, index_ttc_s))", "source_columns": "center_x,y,vx,vy,heading,length,width", "temporal_availability": "index_time_t0", "missing_rule": "capped at 10.0s"},
        {"feature_name": "is_ttc_finite", "role": "physics", "formula": "1.0 if index_ttc_s < 10.0 else 0.0", "source_columns": "index_ttc_s", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "is_censored", "role": "physics", "formula": "1.0 if index_ttc_s >= 10.0 else 0.0", "source_columns": "index_ttc_s", "temporal_availability": "index_time_t0", "missing_rule": "1.0"},
        {"feature_name": "index_distance_m", "role": "physics", "formula": "sqrt((x_tgt - x_ego)^2 + (y_tgt - y_ego)^2)", "source_columns": "center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "radial_closing_speed_mps", "role": "physics", "formula": "- (rel_v . rel_u)", "source_columns": "velocity_x, velocity_y, center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "ego_longitudinal_pos_m", "role": "physics", "formula": "dx*cos(theta_ego) + dy*sin(theta_ego)", "source_columns": "center_x, center_y, heading", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "ego_lateral_pos_m", "role": "physics", "formula": "-dx*sin(theta_ego) + dy*cos(theta_ego)", "source_columns": "center_x, center_y, heading", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "ego_longitudinal_vel_mps", "role": "physics", "formula": "dvx*cos(theta_ego) + dvy*sin(theta_ego)", "source_columns": "velocity_x, velocity_y, heading", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "ego_lateral_vel_mps", "role": "physics", "formula": "-dvx*sin(theta_ego) + dvy*cos(theta_ego)", "source_columns": "velocity_x, velocity_y, heading", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "ego_speed_mps", "role": "physics", "formula": "sqrt(vx_ego^2 + vy_ego^2)", "source_columns": "velocity_x, velocity_y", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "target_speed_mps", "role": "physics", "formula": "sqrt(vx_tgt^2 + vy_tgt^2)", "source_columns": "velocity_x, velocity_y", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "rel_heading_sin", "role": "physics", "formula": "sin(theta_tgt - theta_ego)", "source_columns": "heading", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "rel_heading_cos", "role": "physics", "formula": "cos(theta_tgt - theta_ego)", "source_columns": "heading", "temporal_availability": "index_time_t0", "missing_rule": "required finite"},
        {"feature_name": "ego_accel_mps2", "role": "physics", "formula": "derived_accel_mps2 from history t<=10", "source_columns": "derived_accel_mps2", "temporal_availability": "history_t<=10", "missing_rule": "0.0 if missing"},
        {"feature_name": "ego_accel_valid", "role": "physics", "formula": "1.0 if derived_accel valid else 0.0", "source_columns": "derived_accel_mps2", "temporal_availability": "history_t<=10", "missing_rule": "indicator"},
        {"feature_name": "target_accel_mps2", "role": "physics", "formula": "derived_accel_mps2 from history t<=10", "source_columns": "derived_accel_mps2", "temporal_availability": "history_t<=10", "missing_rule": "0.0 if missing"},
        {"feature_name": "target_accel_valid", "role": "physics", "formula": "1.0 if derived_accel valid else 0.0", "source_columns": "derived_accel_mps2", "temporal_availability": "history_t<=10", "missing_rule": "indicator"},
        {"feature_name": "ego_length_m", "role": "physics", "formula": "length", "source_columns": "length", "temporal_availability": "index_time_t0", "missing_rule": "required > 0"},
        {"feature_name": "ego_width_m", "role": "physics", "formula": "width", "source_columns": "width", "temporal_availability": "index_time_t0", "missing_rule": "required > 0"},
        {"feature_name": "target_length_m", "role": "physics", "formula": "length", "source_columns": "length", "temporal_availability": "index_time_t0", "missing_rule": "required > 0"},
        {"feature_name": "target_width_m", "role": "physics", "formula": "width", "source_columns": "width", "temporal_availability": "index_time_t0", "missing_rule": "required > 0"},
        
        # Context Core
        {"feature_name": "n_actors_30m", "role": "context", "formula": "count(3rd_party where dist<=30m)", "source_columns": "center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "n_actors_50m", "role": "context", "formula": "count(3rd_party where dist<=50m)", "source_columns": "center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "n_actors_70m", "role": "context", "formula": "count(3rd_party where dist<=70m)", "source_columns": "center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "n_vehicles_70m", "role": "context", "formula": "count(3rd_party vehicles where dist<=70m)", "source_columns": "object_type, center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "n_pedestrians_70m", "role": "context", "formula": "count(3rd_party pedestrians where dist<=70m)", "source_columns": "object_type, center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "n_cyclists_70m", "role": "context", "formula": "count(3rd_party cyclists where dist<=70m)", "source_columns": "object_type, center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0"},
        {"feature_name": "nearest_third_party_dist_m", "role": "context", "formula": "min(dist to 3rd_party actors)", "source_columns": "center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "70.0 if empty"},
        {"feature_name": "mean_third_party_speed_mps", "role": "context", "formula": "mean(speeds of 3rd_party actors)", "source_columns": "velocity_x, velocity_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0 if empty"},
        {"feature_name": "std_third_party_speed_mps", "role": "context", "formula": "std(speeds of 3rd_party actors)", "source_columns": "velocity_x, velocity_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0 if empty"},
        {"feature_name": "circular_heading_dispersion", "role": "context", "formula": "1 - sqrt(mean(cos)^2 + mean(sin)^2)", "source_columns": "heading", "temporal_availability": "index_time_t0", "missing_rule": "0.0 if empty"},
        {"feature_name": "max_positive_closing_pressure", "role": "context", "formula": "max_k max(0, -rel_v_k . u_k)", "source_columns": "velocity_x, velocity_y, center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0 if empty"},
        {"feature_name": "agg_positive_closing_pressure", "role": "context", "formula": "sum_k max(0, closing_k) / max(1.0, dist_k)", "source_columns": "velocity_x, velocity_y, center_x, center_y", "temporal_availability": "index_time_t0", "missing_rule": "0.0 if empty"},
        {"feature_name": "context_empty", "role": "context", "formula": "1.0 if n_actors_70m == 0 else 0.0", "source_columns": "n_actors_70m", "temporal_availability": "index_time_t0", "missing_rule": "indicator"},
    ]
    return pd.DataFrame(dict_records)
