#!/usr/bin/env python3
"""
Scene-Level OBB-TTC Criticality Profiling Engine
================================================
Canonical implementation of the continuous scene-level geometric/kinematic
criticality profile (RQ1).

Key Concepts:
1. Frame Estimand:
   - For all valid targets within 70m of SDC at frame t, compute swept SAT OBB TTC.
   - scene_ttc_min_s(t) = min_i TTC_obb_i(t)
   - dominant_actor_id(t) = argmin_i TTC_obb_i(t)
   - Deterministic tie-break: min boundary clearance -> min center distance -> min track_id
   - Continuous severity: C_t = 1 - (clip(scene_ttc_min_s, 0.0, 10.0) - 0.0) / 10.0
   - Mutually exclusive frame status: event, right_censored, no_exposure, invalid_ego, invalid_frame

2. Scenario Multi-Component Criticality Profile (D_s):
   - valid_frame_fraction, exposure_frame_fraction
   - criticality_peak, criticality_auc_s, criticality_mean_exposed
   - Time Exposed to TTC <= tau: tet_tau_s (tau in 1.0, 2.0, 3.0, 5.0s)
   - Criticality episodes: episode_count_tau, max_episode_duration_tau_s (min_dur=0.5s, merge_gap=0.5s)
   - Recovery time: recovery_time_tau_s, recovery_censored
   - Interaction turnover (Secondary descriptor): dominant_actor_turnover_count, dominant_actor_turnover_rate, dominant_actor_type_composition
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

from phase2_womd.obb_ttc_swept import OBBAgent, SweptTTCResult, compute_ttc_obb_swept
from phase2_womd.schema import (
    INDEX_TIME_RADIUS_M,
    TTC_CRITICAL_THRESHOLD_S,
)


@dataclass
class FrameSceneCriticality:
    """Frame-level scene criticality outcome at time index t."""
    scenario_id: str
    time_index: int
    timestamp_s: float
    status: str  # "current_geometry_overlap" | "future_contact_event" | "right_censored" | "no_exposure" | "invalid_ego_state" | "invalid_frame"
    
    # Criticality values
    scene_ttc_min_s: float = float("nan")
    severity_c_t: float = float("nan")
    is_critical_3s: bool = False
    censor_time_s: float = 10.0
    
    # Boolean status indicators
    is_exposed: bool = False
    is_event: bool = False
    is_censored: bool = False
    is_overlap: bool = False
    
    # Outcome-Independent Focal Actor (Nearest True-OBB-Clearance)
    focal_actor_id_nearest_clearance: Optional[int] = None
    focal_actor_type: Optional[str] = None
    focal_clearance_m: float = float("nan")
    focal_center_distance_m: float = float("nan")
    focal_closing_speed_mps: float = float("nan")
    
    # Outcome-Conditioned Dominant Actor (Minimum TTC — for sensitivity audit only)
    dominant_actor_id_ttc_min: Optional[int] = None
    dominant_actor_type: Optional[str] = None
    dominant_clearance_m: float = float("nan")
    dominant_center_distance_m: float = float("nan")
    dominant_closing_speed_mps: float = float("nan")
    
    # Exposure context
    n_valid_targets_70m: int = 0
    n_vehicles_70m: int = 0
    n_pedestrians_70m: int = 0
    n_cyclists_70m: int = 0


@dataclass
class ScenarioCriticalityProfile:
    """Multi-component scenario criticality profile D_s across the full 9-second run."""
    scenario_id: str
    split: str
    total_frames: int
    valid_frames: int
    exposed_frames: int
    
    valid_frame_fraction: float
    exposure_frame_fraction: float
    
    # Continuous Severity Metrics
    criticality_peak: float
    criticality_auc_s: float
    criticality_mean_exposed: float
    
    # Time-Exposed-to-TTC (TET) Metrics
    tet_1s_s: float
    tet_2s_s: float
    tet_3s_s: float
    tet_5s_s: float
    
    # Episode Analysis (Primary tau = 3.0s, min_duration=0.5s, merge_gap=0.5s)
    episode_count_3s: int
    max_episode_duration_3s_s: float
    recovery_time_3s_s: float
    recovery_censored_3s: bool
    
    # Secondary Interaction Composition & Turnover
    dominant_actor_turnover_count: int
    dominant_actor_turnover_rate: float
    fraction_dominant_vehicle: float
    fraction_dominant_pedestrian: float
    fraction_dominant_cyclist: float


def _parse_obb_agent(row: Any) -> Optional[OBBAgent]:
    """Parse OBBAgent with full geometry and finite validation."""
    try:
        valid = bool(row.get("valid", True) if isinstance(row, dict) else row["valid"])
        if not valid:
            return None
        cx = float(row.get("center_x") if isinstance(row, dict) else row["center_x"])
        cy = float(row.get("center_y") if isinstance(row, dict) else row["center_y"])
        vx = float(row.get("velocity_x") if isinstance(row, dict) else row["velocity_x"])
        vy = float(row.get("velocity_y") if isinstance(row, dict) else row["velocity_y"])
        heading = float(row.get("heading") if isinstance(row, dict) else row["heading"])
        length = float(row.get("length") if isinstance(row, dict) else row["length"])
        width = float(row.get("width") if isinstance(row, dict) else row["width"])
        
        if (math.isnan(cx) or math.isnan(cy) or math.isnan(vx) or math.isnan(vy) or
            math.isnan(heading) or math.isnan(length) or math.isnan(width) or
            length <= 0 or width <= 0):
            return None
            
        return OBBAgent(
            cx=cx, cy=cy, length=length, width=width,
            heading=heading, vx=vx, vy=vy, valid=True
        )
    except Exception:
        return None


def compute_frame_scene_criticality(
    sdc_row: Optional[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    scenario_id: str,
    time_index: int,
    timestamp_s: float,
    radius_m: float = 70.0,
    t_min_s: float = 0.0,
    t_max_s: float = 10.0,
    tau_critical_s: float = 3.0,
) -> FrameSceneCriticality:
    """
    Compute ground-truth OBB-TTC reference outcome for current frame.
    Strictly past/present states only — zero future trajectory leakage.
    Uses exact convex OBB boundary clearance and 6-state taxonomy.
    """
    from phase2_womd.obb_ttc_swept import compute_obb_boundary_clearance
    
    if sdc_row is None:
        return FrameSceneCriticality(
            scenario_id=scenario_id,
            time_index=time_index,
            timestamp_s=timestamp_s,
            status="invalid_ego_state",
            censor_time_s=t_max_s,
        )
        
    ego_agent = _parse_obb_agent(sdc_row)
    if ego_agent is None:
        return FrameSceneCriticality(
            scenario_id=scenario_id,
            time_index=time_index,
            timestamp_s=timestamp_s,
            status="invalid_ego_state",
            censor_time_s=t_max_s,
        )
        
    candidate_evaluations = []
    n_veh, n_ped, n_cyc = 0, 0, 0
    
    for trow in target_rows:
        raw_type = str(trow.get("object_type", "") if isinstance(trow, dict) else trow["object_type"])
        if raw_type not in ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"):
            continue
            
        tgt_agent = _parse_obb_agent(trow)
        if tgt_agent is None:
            continue
            
        dx = tgt_agent.cx - ego_agent.cx
        dy = tgt_agent.cy - ego_agent.cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > radius_m:
            continue
            
        if raw_type == "TYPE_VEHICLE":
            n_veh += 1
        elif raw_type == "TYPE_PEDESTRIAN":
            n_ped += 1
        elif raw_type == "TYPE_CYCLIST":
            n_cyc += 1
            
        # Radial closing speed
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
            rel_vx = tgt_agent.vx - ego_agent.vx
            rel_vy = tgt_agent.vy - ego_agent.vy
            cl_spd = -(rel_vx * ux + rel_vy * uy)
        else:
            cl_spd = 0.0
            
        # Exact True OBB Boundary Clearance
        exact_clearance = compute_obb_boundary_clearance(ego_agent, tgt_agent)
        if math.isnan(exact_clearance):
            exact_clearance = max(0.0, dist - 0.5 * (ego_agent.length + tgt_agent.length))
            
        # Swept SAT OBB TTC
        ttc_res = compute_ttc_obb_swept(ego_agent, tgt_agent, t_max=t_max_s)
        
        if ttc_res.overlap_now:
            val_ttc = 0.0
            tgt_status = "current_geometry_overlap"
        elif ttc_res.hit_future:
            val_ttc = max(0.0, ttc_res.ttc_s)
            tgt_status = "future_contact_event"
        else:
            val_ttc = t_max_s
            tgt_status = "right_censored"
            
        tid = int(trow["track_id"])
        candidate_evaluations.append((val_ttc, exact_clearance, dist, tid, raw_type, cl_spd, tgt_status))
        
    n_targets = len(candidate_evaluations)
    if n_targets == 0:
        return FrameSceneCriticality(
            scenario_id=scenario_id,
            time_index=time_index,
            timestamp_s=timestamp_s,
            status="no_exposure",
            scene_ttc_min_s=float("nan"),
            severity_c_t=float("nan"),
            censor_time_s=t_max_s,
            is_exposed=False,
            is_event=False,
            is_censored=False,
            is_overlap=False,
            n_valid_targets_70m=0,
            n_vehicles_70m=0,
            n_pedestrians_70m=0,
            n_cyclists_70m=0,
        )
        
    # 1. Outcome-Independent Focal Actor: Select by (exact_clearance asc, dist asc, tid asc)
    eval_by_clearance = sorted(candidate_evaluations, key=lambda item: (item[1], item[2], item[3]))
    focal = eval_by_clearance[0]
    _, focal_clr, focal_dst, focal_tid, focal_type, focal_cl_spd, _ = focal
    
    # 2. Outcome-Conditioned Reference Target: Select by (TTC asc, clearance asc, dist asc, tid asc)
    eval_by_ttc = sorted(candidate_evaluations, key=lambda item: (item[0], item[1], item[2], item[3]))
    best = eval_by_ttc[0]
    min_ttc, best_clr, best_dst, best_tid, best_type, best_cl_spd, best_tgt_status = best
    
    # Frame Status Determination
    if best_tgt_status == "current_geometry_overlap" or min_ttc <= 1e-6:
        frame_status = "current_geometry_overlap"
        c_t = 1.0
        is_overlap = True
        is_event = True
        is_censored = False
    elif best_tgt_status == "future_contact_event" and min_ttc < t_max_s:
        frame_status = "future_contact_event"
        c_t = 1.0 - (min_ttc - t_min_s) / (t_max_s - t_min_s)
        is_overlap = False
        is_event = True
        is_censored = False
    else:
        frame_status = "right_censored"
        min_ttc = t_max_s
        c_t = 0.0
        is_overlap = False
        is_event = False
        is_censored = True
        
    c_t = float(max(0.0, min(1.0, c_t)))
    
    return FrameSceneCriticality(
        scenario_id=scenario_id,
        time_index=time_index,
        timestamp_s=timestamp_s,
        status=frame_status,
        scene_ttc_min_s=float(min_ttc),
        severity_c_t=float(c_t),
        is_critical_3s=bool(min_ttc <= tau_critical_s),
        censor_time_s=t_max_s,
        is_exposed=True,
        is_event=is_event,
        is_censored=is_censored,
        is_overlap=is_overlap,
        focal_actor_id_nearest_clearance=focal_tid,
        focal_actor_type=focal_type,
        focal_clearance_m=float(focal_clr),
        focal_center_distance_m=float(focal_dst),
        focal_closing_speed_mps=float(focal_cl_spd),
        dominant_actor_id_ttc_min=best_tid,
        dominant_actor_type=best_type,
        dominant_clearance_m=float(best_clr),
        dominant_center_distance_m=float(best_dst),
        dominant_closing_speed_mps=float(best_cl_spd),
        n_valid_targets_70m=n_targets,
        n_vehicles_70m=n_veh,
        n_pedestrians_70m=n_ped,
        n_cyclists_70m=n_cyc,
    )


def extract_scenario_criticality_profile(
    frame_criticalities: List[FrameSceneCriticality],
    scenario_id: str,
    split: str = "train",
    tau_primary_s: float = 3.0,
    min_episode_duration_s: float = 0.5,
    merge_gap_s: float = 0.5,
    recovery_hold_time_s: float = 0.5,
) -> ScenarioCriticalityProfile:
    """
    Compute multi-component scenario criticality profile D_s from continuous frames.
    """
    total_frames = len(frame_criticalities)
    if total_frames == 0:
        return ScenarioCriticalityProfile(
            scenario_id=scenario_id, split=split, total_frames=0, valid_frames=0, exposed_frames=0,
            valid_frame_fraction=0.0, exposure_frame_fraction=0.0, criticality_peak=0.0, criticality_auc_s=0.0,
            criticality_mean_exposed=0.0, tet_1s_s=0.0, tet_2s_s=0.0, tet_3s_s=0.0, tet_5s_s=0.0,
            episode_count_3s=0, max_episode_duration_3s_s=0.0, recovery_time_3s_s=0.0, recovery_censored_3s=True,
            dominant_actor_turnover_count=0, dominant_actor_turnover_rate=0.0,
            fraction_dominant_vehicle=0.0, fraction_dominant_pedestrian=0.0, fraction_dominant_cyclist=0.0,
        )
        
    valid_frames = [f for f in frame_criticalities if f.status not in ("invalid_ego_state", "invalid_frame")]
    exposed_frames = [f for f in valid_frames if f.is_exposed]
    
    n_valid = len(valid_frames)
    n_exposed = len(exposed_frames)
    
    val_frac = float(n_valid / total_frames) if total_frames > 0 else 0.0
    exp_frac = float(n_exposed / n_valid) if n_valid > 0 else 0.0
    
    if n_exposed == 0:
        return ScenarioCriticalityProfile(
            scenario_id=scenario_id, split=split, total_frames=total_frames, valid_frames=n_valid, exposed_frames=0,
            valid_frame_fraction=val_frac, exposure_frame_fraction=0.0, criticality_peak=0.0, criticality_auc_s=0.0,
            criticality_mean_exposed=0.0, tet_1s_s=0.0, tet_2s_s=0.0, tet_3s_s=0.0, tet_5s_s=0.0,
            episode_count_3s=0, max_episode_duration_3s_s=0.0, recovery_time_3s_s=0.0, recovery_censored_3s=True,
            dominant_actor_turnover_count=0, dominant_actor_turnover_rate=0.0,
            fraction_dominant_vehicle=0.0, fraction_dominant_pedestrian=0.0, fraction_dominant_cyclist=0.0,
        )
        
    # Severity aggregations
    c_vals = [f.severity_c_t for f in exposed_frames if not math.isnan(f.severity_c_t)]
    crit_peak = float(max(c_vals)) if c_vals else 0.0
    crit_mean = float(np.mean(c_vals)) if c_vals else 0.0
    
    # AUC: sum(C_t * dt) across continuous frames
    auc_s = 0.0
    for i in range(len(frame_criticalities)):
        fc = frame_criticalities[i]
        if not math.isnan(fc.severity_c_t):
            # dt is difference to next frame or standard 0.1s
            if i + 1 < len(frame_criticalities):
                dt = max(0.01, min(0.5, frame_criticalities[i+1].timestamp_s - fc.timestamp_s))
            else:
                dt = 0.1
            auc_s += fc.severity_c_t * dt
            
    # TET calculations (Time Exposed to TTC <= tau)
    def _compute_tet(tau: float) -> float:
        tot_dur = 0.0
        for i, fc in enumerate(frame_criticalities):
            if not math.isnan(fc.scene_ttc_min_s) and fc.scene_ttc_min_s <= tau:
                if i + 1 < len(frame_criticalities):
                    dt = max(0.01, min(0.5, frame_criticalities[i+1].timestamp_s - fc.timestamp_s))
                else:
                    dt = 0.1
                tot_dur += dt
        return float(tot_dur)
        
    tet_1s = _compute_tet(1.0)
    tet_2s = _compute_tet(2.0)
    tet_3s = _compute_tet(3.0)
    tet_5s = _compute_tet(5.0)
    
    # Episode analysis for primary tau=3.0s
    raw_critical_intervals: List[Tuple[float, float]] = []
    in_interval = False
    start_t = 0.0
    last_t = 0.0
    
    for fc in frame_criticalities:
        is_crit = (not math.isnan(fc.scene_ttc_min_s) and fc.scene_ttc_min_s <= tau_primary_s)
        if is_crit:
            if not in_interval:
                in_interval = True
                start_t = fc.timestamp_s
            last_t = fc.timestamp_s
        else:
            if in_interval:
                in_interval = False
                raw_critical_intervals.append((start_t, last_t + 0.1))
    if in_interval:
        raw_critical_intervals.append((start_t, last_t + 0.1))
        
    # Merge gaps <= merge_gap_s
    merged_episodes: List[Tuple[float, float]] = []
    for interval in raw_critical_intervals:
        if not merged_episodes:
            merged_episodes.append(interval)
        else:
            prev_start, prev_end = merged_episodes[-1]
            if interval[0] - prev_end <= merge_gap_s:
                merged_episodes[-1] = (prev_start, max(prev_end, interval[1]))
            else:
                merged_episodes.append(interval)
                
    # Filter by min duration >= min_episode_duration_s
    valid_episodes = [ep for ep in merged_episodes if (ep[1] - ep[0]) >= min_episode_duration_s]
    episode_count = len(valid_episodes)
    max_ep_duration = max([(ep[1] - ep[0]) for ep in valid_episodes]) if valid_episodes else 0.0
    
    # Recovery time analysis from peak severity
    peak_idx = int(np.argmax(c_vals)) if c_vals else 0
    peak_time = exposed_frames[peak_idx].timestamp_s if exposed_frames else 0.0
    recovery_time = 0.0
    recovery_censored = True
    
    # Search for safe hold (TTC > tau for recovery_hold_time_s) after peak_time
    safe_streak_dur = 0.0
    found_recovery = False
    for fc in frame_criticalities:
        if fc.timestamp_s < peak_time:
            continue
        is_safe = (math.isnan(fc.scene_ttc_min_s) or fc.scene_ttc_min_s > tau_primary_s)
        if is_safe:
            safe_streak_dur += 0.1
            if safe_streak_dur >= recovery_hold_time_s:
                recovery_time = (fc.timestamp_s - recovery_hold_time_s) - peak_time
                recovery_censored = False
                found_recovery = True
                break
        else:
            safe_streak_dur = 0.0
            
    if not found_recovery:
        recovery_time = frame_criticalities[-1].timestamp_s - peak_time if frame_criticalities else 0.0
        recovery_censored = True
        
    # Interaction composition & Dominant actor turnover
    turnovers = 0
    prev_dominant = None
    veh_count, ped_count, cyc_count = 0, 0, 0
    
    for fc in exposed_frames:
        dom_id = fc.dominant_actor_id_ttc_min
        if dom_id is not None:
            if prev_dominant is not None and dom_id != prev_dominant:
                turnovers += 1
            prev_dominant = dom_id
            
        if fc.dominant_actor_type == "TYPE_VEHICLE":
            veh_count += 1
        elif fc.dominant_actor_type == "TYPE_PEDESTRIAN":
            ped_count += 1
        elif fc.dominant_actor_type == "TYPE_CYCLIST":
            cyc_count += 1
            
    turnover_rate = float(turnovers / n_exposed) if n_exposed > 0 else 0.0
    frac_veh = float(veh_count / n_exposed) if n_exposed > 0 else 0.0
    frac_ped = float(ped_count / n_exposed) if n_exposed > 0 else 0.0
    frac_cyc = float(cyc_count / n_exposed) if n_exposed > 0 else 0.0
    
    return ScenarioCriticalityProfile(
        scenario_id=scenario_id,
        split=split,
        total_frames=total_frames,
        valid_frames=n_valid,
        exposed_frames=n_exposed,
        valid_frame_fraction=val_frac,
        exposure_frame_fraction=exp_frac,
        criticality_peak=crit_peak,
        criticality_auc_s=float(auc_s),
        criticality_mean_exposed=crit_mean,
        tet_1s_s=tet_1s,
        tet_2s_s=tet_2s,
        tet_3s_s=tet_3s,
        tet_5s_s=tet_5s,
        episode_count_3s=episode_count,
        max_episode_duration_3s_s=float(max_ep_duration),
        recovery_time_3s_s=float(max(0.0, recovery_time)),
        recovery_censored_3s=recovery_censored,
        dominant_actor_turnover_count=turnovers,
        dominant_actor_turnover_rate=turnover_rate,
        fraction_dominant_vehicle=frac_veh,
        fraction_dominant_pedestrian=frac_ped,
        fraction_dominant_cyclist=frac_cyc,
    )
