#!/usr/bin/env python3
"""
WACV 2027 ODD Feature Extraction Engine (Phase 2C Repaired)
===========================================================
Extracts structured features across four semantic groups:
1. P_CLEAN_KINEMATIC_CONTROL (12 features):
   - SDC kinematics (speed, tangential accel, yaw rate)
   - Outcome-independent focal actor (nearest true-OBB-clearance) relative pos, vel, speed, clearance, type.
   - ZERO direct target leakage (TTC, C_t, hit flags excluded).
2. E_ODD_STATIC_INFRASTRUCTURE (7 features):
   - Static map geometry (crosswalk, stop sign, speed bump, road edge, lane point density, lane heading dispersion, signal stop point).
3. E_ODD_DYNAMIC_CONTEXT (13 features):
   - Actor counts in 30/50/70m, vehicle/vulnerable proportions, Shannon entropy.
   - Outcome-independent third-party (excluding SDC and nearest-clearance focal actor) distance, speed, closing pressure max/sum, active signals.
4. E_ODD_HISTORY (7 features):
   - Strictly past-only rolling 1.0s statistics over u in [max(0, t-10), t].
   - Actual past-frame dynamic closing pressure rolling mean, max, slope.
   - Actor turnover over valid non-SDC tracks and history coverage indicator.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd

from phase2_womd.obb_ttc_swept import OBBAgent, compute_obb_boundary_clearance
from phase2_womd.scene_criticality_engine import FrameSceneCriticality, _parse_obb_agent


# ---------------------------------------------------------------------------
# Feature Names Registry
# ---------------------------------------------------------------------------

P_CLEAN_FEATURE_NAMES = [
    "p_clean__sdc_speed_mps",
    "p_clean__sdc_accel_mps2",
    "p_clean__sdc_yaw_rate_radps",
    "p_clean__focal_rel_long_pos_m",
    "p_clean__focal_rel_lat_pos_m",
    "p_clean__focal_rel_long_vel_mps",
    "p_clean__focal_rel_lat_vel_mps",
    "p_clean__focal_speed_mps",
    "p_clean__focal_clearance_m",
    "p_clean__focal_center_distance_m",
    "p_clean__focal_is_vehicle",
    "p_clean__focal_is_vulnerable",
]

E_STATIC_FEATURE_NAMES = [
    "e_static__dist_nearest_crosswalk_m",
    "e_static__dist_nearest_stop_sign_m",
    "e_static__dist_nearest_speed_bump_m",
    "e_static__dist_nearest_road_edge_m",
    "e_static__lane_point_density_50m",
    "e_static__lane_heading_dispersion_50m",
    "e_static__dist_nearest_signal_stop_point_m",
]

E_DYNAMIC_FEATURE_NAMES = [
    "e_dynamic__n_actors_30m",
    "e_dynamic__n_actors_50m",
    "e_dynamic__n_actors_70m",
    "e_dynamic__vehicle_proportion_70m",
    "e_dynamic__vulnerable_proportion_70m",
    "e_dynamic__actor_type_entropy_70m",
    "e_dynamic__third_party_nearest_dist_m",
    "e_dynamic__third_party_mean_speed_mps",
    "e_dynamic__third_party_speed_std_mps",
    "e_dynamic__third_party_heading_dispersion",
    "e_dynamic__third_party_closing_pressure_max",
    "e_dynamic__third_party_closing_pressure_sum",
    "e_dynamic__active_stop_signal_count_70m",
]

E_HIST_FEATURE_NAMES = [
    "e_hist__n_actors_70m_mean_1s",
    "e_hist__n_actors_70m_std_1s",
    "e_hist__n_actors_70m_slope_1s",
    "e_hist__closing_pressure_sum_mean_1s",
    "e_hist__closing_pressure_sum_max_1s",
    "e_hist__closing_pressure_sum_slope_1s",
    "e_hist__actor_composition_turnover_1s",
]

ENVIRONMENT_CANDIDATE_FEATURE_NAMES = (
    E_STATIC_FEATURE_NAMES
    + E_DYNAMIC_FEATURE_NAMES
    + E_HIST_FEATURE_NAMES
)

ALL_ODD_FEATURE_NAMES = (
    P_CLEAN_FEATURE_NAMES
    + ENVIRONMENT_CANDIDATE_FEATURE_NAMES
)


# ---------------------------------------------------------------------------
# Map and Signal Spatial Index Helper
# ---------------------------------------------------------------------------

class MapSpatialIndex:
    """Fast spatial index for static map features within a scenario."""
    def __init__(self, map_df: Optional[pd.DataFrame]):
        self.crosswalk_pts = np.empty((0, 2), dtype=np.float64)
        self.stop_sign_pts = np.empty((0, 2), dtype=np.float64)
        self.speed_bump_pts = np.empty((0, 2), dtype=np.float64)
        self.road_edge_pts = np.empty((0, 2), dtype=np.float64)
        self.lane_pts = np.empty((0, 2), dtype=np.float64)
        self.lane_headings = np.empty((0,), dtype=np.float64)
        self.has_map_data = False

        if map_df is not None and not map_df.empty:
            valid_m = map_df["x"].notnull() & map_df["y"].notnull()
            df_v = map_df[valid_m]

            if not df_v.empty:
                self.has_map_data = True

            # Crosswalks
            cw = df_v[df_v["feature_type"] == "crosswalk"]
            if not cw.empty:
                self.crosswalk_pts = cw[["x", "y"]].to_numpy(dtype=np.float64)

            # Stop signs
            ss = df_v[df_v["feature_type"] == "stop_sign"]
            if not ss.empty:
                self.stop_sign_pts = ss[["x", "y"]].to_numpy(dtype=np.float64)

            # Speed bumps
            sb = df_v[df_v["feature_type"] == "speed_bump"]
            if not sb.empty:
                self.speed_bump_pts = sb[["x", "y"]].to_numpy(dtype=np.float64)

            # Road edges
            re = df_v[df_v["feature_type"] == "road_edge"]
            if not re.empty:
                self.road_edge_pts = re[["x", "y"]].to_numpy(dtype=np.float64)

            # Lanes
            lanes = df_v[df_v["feature_type"] == "lane"]
            if not lanes.empty:
                self.lane_pts = lanes[["x", "y"]].to_numpy(dtype=np.float64)
                lane_h_list = []
                for fid, group in lanes.groupby("feature_id", sort=False):
                    pts = group.sort_values("point_index")[["x", "y"]].to_numpy(dtype=np.float64)
                    if len(pts) > 1:
                        dx = np.diff(pts[:, 0])
                        dy = np.diff(pts[:, 1])
                        hs = np.arctan2(dy, dx)
                        lane_h_list.extend(hs)
                        lane_h_list.append(hs[-1])
                    elif len(pts) == 1:
                        lane_h_list.append(0.0)
                if lane_h_list and len(lane_h_list) == len(self.lane_pts):
                    self.lane_headings = np.array(lane_h_list, dtype=np.float64)
                elif self.lane_pts.shape[0] > 0:
                    self.lane_headings = np.zeros(self.lane_pts.shape[0], dtype=np.float64)


def _min_dist_to_points(x0: float, y0: float, pts: np.ndarray, default_dist: float = 100.0) -> float:
    if pts.shape[0] == 0:
        return default_dist
    dx = pts[:, 0] - x0
    dy = pts[:, 1] - y0
    d2 = dx * dx + dy * dy
    return float(math.sqrt(np.min(d2)))


def _compute_frame_third_party_closing_pressure(
    u_idx: int,
    time_agents: Dict[int, Dict[int, Dict[str, Any]]],
    ego_obb: OBBAgent,
    focal_id: Optional[int],
) -> Tuple[float, float]:
    """Compute (max_cp, sum_cp) at time u_idx strictly for third-party actors."""
    u_dict = time_agents.get(u_idx, {})
    pressures = []

    for tid, row in u_dict.items():
        if row.get("is_sdc", False) or row.get("is_sdc") == 1:
            continue
        if focal_id is not None and tid == focal_id:
            continue
        a_obb = _parse_obb_agent(row)
        if a_obb is None:
            continue
        dx = a_obb.cx - ego_obb.cx
        dy = a_obb.cy - ego_obb.cy
        d = math.sqrt(dx * dx + dy * dy)
        if d <= 70.0 and d > 1e-4:
            ux, uy = dx / d, dy / d
            rvx = a_obb.vx - ego_obb.vx
            rvy = a_obb.vy - ego_obb.vy
            cl_spd = -(rvx * ux + rvy * uy)
            pos_cl = max(0.0, cl_spd)
            pressures.append((pos_cl, d))

    if not pressures:
        return 0.0, 0.0

    max_cp = max([p[0] for p in pressures])
    sum_cp = sum([p[0] / max(1.0, p[1]) for p in pressures])
    return float(max_cp), float(sum_cp)


# ---------------------------------------------------------------------------
# Feature Extraction per Frame
# ---------------------------------------------------------------------------

def extract_frame_odd_features(
    time_idx: int,
    time_agents: Dict[int, Dict[int, Dict[str, Any]]],
    frame_crit_lookup: Dict[int, FrameSceneCriticality],
    map_index: Optional[MapSpatialIndex] = None,
    dynamic_signals_by_time: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    max_history_steps: int = 10,
) -> Dict[str, float]:
    """
    Extract clean P, static ODD, dynamic ODD, and past temporal ODD features for frame time_idx.
    Strictly past/present only: u <= time_idx.
    """
    feats: Dict[str, float] = {}

    # Initialize all with NaN
    for f in ALL_ODD_FEATURE_NAMES:
        feats[f] = float("nan")

    fc_t = frame_crit_lookup.get(time_idx)
    t_agents = time_agents.get(time_idx, {})

    sdc_row = None
    for tid, a in t_agents.items():
        if a.get("is_sdc", False) or a.get("is_sdc") == 1:
            sdc_row = a
            break

    if fc_t is None or sdc_row is None:
        return feats

    ego_obb = _parse_obb_agent(sdc_row)
    if ego_obb is None:
        return feats

    ego_spd = math.sqrt(ego_obb.vx ** 2 + ego_obb.vy ** 2)

    # -----------------------------------------------------------------------
    # 1. Group P_CLEAN_KINEMATIC_CONTROL (Outcome-Independent Focal Actor!)
    # -----------------------------------------------------------------------
    feats["p_clean__sdc_speed_mps"] = float(ego_spd)
    feats["p_clean__sdc_accel_mps2"] = float(sdc_row.get("derived_accel_mps2", 0.0) if sdc_row.get("derived_accel_mps2") is not None and not math.isnan(float(sdc_row.get("derived_accel_mps2", 0.0))) else 0.0)
    feats["p_clean__sdc_yaw_rate_radps"] = float(sdc_row.get("derived_yaw_rate_rps", 0.0) if sdc_row.get("derived_yaw_rate_rps") is not None and not math.isnan(float(sdc_row.get("derived_yaw_rate_rps", 0.0))) else 0.0)

    focal_id = fc_t.focal_actor_id_nearest_clearance
    feats["p_clean__focal_clearance_m"] = float(fc_t.focal_clearance_m if not math.isnan(fc_t.focal_clearance_m) else 70.0)
    feats["p_clean__focal_center_distance_m"] = float(fc_t.focal_center_distance_m if not math.isnan(fc_t.focal_center_distance_m) else 70.0)

    if focal_id is not None and focal_id in t_agents:
        focal_row = t_agents[focal_id]
        focal_obb = _parse_obb_agent(focal_row)
        if focal_obb is not None:
            dx = focal_obb.cx - ego_obb.cx
            dy = focal_obb.cy - ego_obb.cy
            cos_h = math.cos(ego_obb.heading)
            sin_h = math.sin(ego_obb.heading)
            feats["p_clean__focal_rel_long_pos_m"] = float(dx * cos_h + dy * sin_h)
            feats["p_clean__focal_rel_lat_pos_m"] = float(-dx * sin_h + dy * cos_h)

            dvx = focal_obb.vx - ego_obb.vx
            dvy = focal_obb.vy - ego_obb.vy
            feats["p_clean__focal_rel_long_vel_mps"] = float(dvx * cos_h + dvy * sin_h)
            feats["p_clean__focal_rel_lat_vel_mps"] = float(-dvx * sin_h + dvy * cos_h)
            feats["p_clean__focal_speed_mps"] = float(math.sqrt(focal_obb.vx ** 2 + focal_obb.vy ** 2))
        else:
            feats["p_clean__focal_rel_long_pos_m"] = 0.0
            feats["p_clean__focal_rel_lat_pos_m"] = 0.0
            feats["p_clean__focal_rel_long_vel_mps"] = 0.0
            feats["p_clean__focal_rel_lat_vel_mps"] = 0.0
            feats["p_clean__focal_speed_mps"] = 0.0
    else:
        feats["p_clean__focal_rel_long_pos_m"] = 0.0
        feats["p_clean__focal_rel_lat_pos_m"] = 0.0
        feats["p_clean__focal_rel_long_vel_mps"] = 0.0
        feats["p_clean__focal_rel_lat_vel_mps"] = 0.0
        feats["p_clean__focal_speed_mps"] = 0.0

    feats["p_clean__focal_is_vehicle"] = 1.0 if fc_t.focal_actor_type == "TYPE_VEHICLE" else 0.0
    feats["p_clean__focal_is_vulnerable"] = 1.0 if fc_t.focal_actor_type in ("TYPE_PEDESTRIAN", "TYPE_CYCLIST") else 0.0

    # -----------------------------------------------------------------------
    # 2. Group E_ODD_STATIC_INFRASTRUCTURE
    # -----------------------------------------------------------------------
    if map_index is not None and map_index.has_map_data:
        feats["e_static__dist_nearest_crosswalk_m"] = _min_dist_to_points(ego_obb.cx, ego_obb.cy, map_index.crosswalk_pts, 100.0)
        feats["e_static__dist_nearest_stop_sign_m"] = _min_dist_to_points(ego_obb.cx, ego_obb.cy, map_index.stop_sign_pts, 100.0)
        feats["e_static__dist_nearest_speed_bump_m"] = _min_dist_to_points(ego_obb.cx, ego_obb.cy, map_index.speed_bump_pts, 100.0)
        feats["e_static__dist_nearest_road_edge_m"] = _min_dist_to_points(ego_obb.cx, ego_obb.cy, map_index.road_edge_pts, 100.0)

        # Local lane density and dispersion within 50m
        if map_index.lane_pts.shape[0] > 0:
            dlx = map_index.lane_pts[:, 0] - ego_obb.cx
            dly = map_index.lane_pts[:, 1] - ego_obb.cy
            d_lane2 = dlx * dlx + dly * dly
            lane_mask_50 = d_lane2 <= (50.0 ** 2)
            n_lane_50 = int(np.sum(lane_mask_50))
            feats["e_static__lane_point_density_50m"] = float(n_lane_50)

            if n_lane_50 > 2 and len(map_index.lane_headings) == len(map_index.lane_pts):
                hs_50 = map_index.lane_headings[lane_mask_50]
                m_cos = np.mean(np.cos(hs_50))
                m_sin = np.mean(np.sin(hs_50))
                r_len = math.sqrt(m_cos ** 2 + m_sin ** 2)
                feats["e_static__lane_heading_dispersion_50m"] = float(max(0.0, min(1.0, 1.0 - r_len)))
            else:
                feats["e_static__lane_heading_dispersion_50m"] = 0.0
        else:
            feats["e_static__lane_point_density_50m"] = 0.0
            feats["e_static__lane_heading_dispersion_50m"] = 0.0
    else:
        feats["e_static__dist_nearest_crosswalk_m"] = 100.0
        feats["e_static__dist_nearest_stop_sign_m"] = 100.0
        feats["e_static__dist_nearest_speed_bump_m"] = 100.0
        feats["e_static__dist_nearest_road_edge_m"] = 100.0
        feats["e_static__lane_point_density_50m"] = 0.0
        feats["e_static__lane_heading_dispersion_50m"] = 0.0

    # Traffic signal stop points
    active_stop_signals = 0
    sig_stop_pts = []
    if dynamic_signals_by_time is not None and time_idx in dynamic_signals_by_time:
        sig_list = dynamic_signals_by_time[time_idx]
        for s in sig_list:
            sx = s.get("stop_point_x")
            sy = s.get("stop_point_y")
            st = str(s.get("signal_state", "")).upper()
            if sx is not None and sy is not None and not math.isnan(sx) and not math.isnan(sy):
                sig_stop_pts.append((sx, sy))
                dx_s = sx - ego_obb.cx
                dy_s = sy - ego_obb.cy
                if (dx_s * dx_s + dy_s * dy_s) <= (70.0 ** 2):
                    if "STOP" in st or "RED" in st:
                        active_stop_signals += 1

    if sig_stop_pts:
        pts_arr = np.array(sig_stop_pts, dtype=np.float64)
        feats["e_static__dist_nearest_signal_stop_point_m"] = _min_dist_to_points(ego_obb.cx, ego_obb.cy, pts_arr, 100.0)
    else:
        feats["e_static__dist_nearest_signal_stop_point_m"] = 100.0

    feats["e_dynamic__active_stop_signal_count_70m"] = float(active_stop_signals)

    # -----------------------------------------------------------------------
    # 3. Group E_ODD_DYNAMIC_CONTEXT (Outcome-Independent Third-Parties!)
    # -----------------------------------------------------------------------
    c30, c50, c70 = 0, 0, 0
    n_veh, n_ped, n_cyc = 0, 0, 0
    third_parties: List[Tuple[OBBAgent, Dict[str, Any]]] = []

    for tid, row in t_agents.items():
        if row.get("is_sdc", False) or row.get("is_sdc") == 1:
            continue
        a_obb = _parse_obb_agent(row)
        if a_obb is None:
            continue
        dx = a_obb.cx - ego_obb.cx
        dy = a_obb.cy - ego_obb.cy
        d = math.sqrt(dx * dx + dy * dy)
        if d <= 70.0:
            c70 += 1
            otype = str(row.get("object_type", ""))
            if "VEHICLE" in otype:
                n_veh += 1
            elif "PEDESTRIAN" in otype:
                n_ped += 1
            elif "CYCLIST" in otype:
                n_cyc += 1
            if d <= 50.0:
                c50 += 1
            if d <= 30.0:
                c30 += 1

            if focal_id is None or tid != focal_id:
                third_parties.append((a_obb, row))

    feats["e_dynamic__n_actors_30m"] = float(c30)
    feats["e_dynamic__n_actors_50m"] = float(c50)
    feats["e_dynamic__n_actors_70m"] = float(c70)

    if c70 > 0:
        feats["e_dynamic__vehicle_proportion_70m"] = float(n_veh / c70)
        feats["e_dynamic__vulnerable_proportion_70m"] = float((n_ped + n_cyc) / c70)
        props = [n_veh / c70, n_ped / c70, n_cyc / c70]
        ent = -sum([p * math.log(p) for p in props if p > 1e-6])
        feats["e_dynamic__actor_type_entropy_70m"] = float(ent)
    else:
        feats["e_dynamic__vehicle_proportion_70m"] = 0.0
        feats["e_dynamic__vulnerable_proportion_70m"] = 0.0
        feats["e_dynamic__actor_type_entropy_70m"] = 0.0

    # Third party kinematics & closing pressure
    tp_dists, tp_speeds, tp_cos, tp_sin = [], [], [], []
    closing_pressures = []

    for tp_obb, tp_row in third_parties:
        dx3 = tp_obb.cx - ego_obb.cx
        dy3 = tp_obb.cy - ego_obb.cy
        d3 = math.sqrt(dx3 * dx3 + dy3 * dy3)
        tp_dists.append(d3)
        spd3 = math.sqrt(tp_obb.vx ** 2 + tp_obb.vy ** 2)
        tp_speeds.append(spd3)
        tp_cos.append(math.cos(tp_obb.heading))
        tp_sin.append(math.sin(tp_obb.heading))

        if d3 > 1e-4:
            ux, uy = dx3 / d3, dy3 / d3
            rvx = tp_obb.vx - ego_obb.vx
            rvy = tp_obb.vy - ego_obb.vy
            cl_spd = -(rvx * ux + rvy * uy)
            pos_cl = max(0.0, cl_spd)
            closing_pressures.append((pos_cl, d3))

    if tp_dists:
        feats["e_dynamic__third_party_nearest_dist_m"] = float(min(tp_dists))
        feats["e_dynamic__third_party_mean_speed_mps"] = float(np.mean(tp_speeds))
        feats["e_dynamic__third_party_speed_std_mps"] = float(np.std(tp_speeds)) if len(tp_speeds) > 1 else 0.0

        m_cos = np.mean(tp_cos)
        m_sin = np.mean(tp_sin)
        r_len = math.sqrt(m_cos ** 2 + m_sin ** 2)
        feats["e_dynamic__third_party_heading_dispersion"] = float(max(0.0, min(1.0, 1.0 - r_len)))

        max_cp = max([p[0] for p in closing_pressures]) if closing_pressures else 0.0
        sum_cp = sum([p[0] / max(1.0, p[1]) for p in closing_pressures]) if closing_pressures else 0.0
        feats["e_dynamic__third_party_closing_pressure_max"] = float(max_cp)
        feats["e_dynamic__third_party_closing_pressure_sum"] = float(sum_cp)
    else:
        feats["e_dynamic__third_party_nearest_dist_m"] = 70.0
        feats["e_dynamic__third_party_mean_speed_mps"] = 0.0
        feats["e_dynamic__third_party_speed_std_mps"] = 0.0
        feats["e_dynamic__third_party_heading_dispersion"] = 0.0
        feats["e_dynamic__third_party_closing_pressure_max"] = 0.0
        feats["e_dynamic__third_party_closing_pressure_sum"] = 0.0

    # -----------------------------------------------------------------------
    # 4. Group E_ODD_HISTORY (Strict Past-Only u in [max(0, t - 10), t])
    # -----------------------------------------------------------------------
    hist_actors_counts = []
    hist_actor_sets = []
    hist_cp_sums = []

    start_u = max(0, time_idx - max_history_steps)
    for u in range(start_u, time_idx + 1):
        fc_u = frame_crit_lookup.get(u)
        if fc_u is not None and not math.isnan(fc_u.n_valid_targets_70m):
            hist_actors_counts.append(fc_u.n_valid_targets_70m)
        else:
            hist_actors_counts.append(0)

        u_agents = time_agents.get(u, {})
        valid_u_tids = {tid for tid, row in u_agents.items() if not (row.get("is_sdc", False) or row.get("is_sdc") == 1)}
        hist_actor_sets.append(valid_u_tids)

        # Actual rolling closing pressure computation across past frames
        _, u_sum_cp = _compute_frame_third_party_closing_pressure(u, time_agents, ego_obb, focal_id)
        hist_cp_sums.append(u_sum_cp)

    feats["e_hist__n_actors_70m_mean_1s"] = float(np.mean(hist_actors_counts)) if hist_actors_counts else float(c70)
    feats["e_hist__n_actors_70m_std_1s"] = float(np.std(hist_actors_counts)) if len(hist_actors_counts) > 1 else 0.0

    if len(hist_actors_counts) > 1:
        dt_span = max(0.1, (len(hist_actors_counts) - 1) * 0.1)
        slope_act = (hist_actors_counts[-1] - hist_actors_counts[0]) / dt_span
        feats["e_hist__n_actors_70m_slope_1s"] = float(slope_act)
    else:
        feats["e_hist__n_actors_70m_slope_1s"] = 0.0

    feats["e_hist__closing_pressure_sum_mean_1s"] = float(np.mean(hist_cp_sums)) if hist_cp_sums else float(feats["e_dynamic__third_party_closing_pressure_sum"])
    feats["e_hist__closing_pressure_sum_max_1s"] = float(np.max(hist_cp_sums)) if hist_cp_sums else float(feats["e_dynamic__third_party_closing_pressure_max"])

    if len(hist_cp_sums) > 1:
        dt_span = max(0.1, (len(hist_cp_sums) - 1) * 0.1)
        slope_cp = (hist_cp_sums[-1] - hist_cp_sums[0]) / dt_span
        feats["e_hist__closing_pressure_sum_slope_1s"] = float(slope_cp)
    else:
        feats["e_hist__closing_pressure_sum_slope_1s"] = 0.0

    turnovers = 0
    for i in range(1, len(hist_actor_sets)):
        if len(hist_actor_sets[i].symmetric_difference(hist_actor_sets[i - 1])) > 0:
            turnovers += 1
    feats["e_hist__actor_composition_turnover_1s"] = float(turnovers)

    return feats
