#!/usr/bin/env python3
"""
2D OBB Time-To-Collision (TTC) Calculator
==========================================
Oriented Bounding Box collision detection using the Separating Axis
Theorem (SAT), with constant-velocity forward projection for TTC.

Assumptions (must be stated explicitly):
  - Constant velocity extrapolation (no acceleration model)
  - Fixed heading geometry (OBB orientation does not change)
  - 2D collision-course proxy (z ignored)
  - Current overlap → TTC = 0
  - Future OBB overlap within horizon → TTC = first overlap time
  - No overlap within horizon → right-censored at T_MAX_S

Does NOT use best_ttci = 1/TTC conversion.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T_MAX_S = 10.0
T_MIN_S = 0.05
DT_STEP = 0.1       # forward projection step (seconds)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AgentBox:
    """Represents an agent's 2D state for OBB TTC calculation."""
    cx: float          # center x
    cy: float          # center y
    length: float      # bounding box length (along heading)
    width: float       # bounding box width  (perpendicular to heading)
    heading: float     # yaw in radians
    vx: float          # velocity x (m/s)
    vy: float          # velocity y (m/s)
    valid: bool = True

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)


@dataclass
class TTCResult:
    """Result of a TTC computation between two agents."""
    derived_ttc_2d_s: float           # TTC value (seconds)
    derived_dtc_m: float              # distance-to-collision (closest approach)
    derived_closing_speed_mps: float  # rate of range decrease
    derived_overlap_now: bool         # current OBBs overlap
    derived_hit_future: bool          # future overlap found within horizon
    derived_pair_valid: bool          # pair is geometrically valid
    derived_ttc_status: str           # "event" | "right_censored"


# ---------------------------------------------------------------------------
# OBB Geometry
# ---------------------------------------------------------------------------

def obb_corners(cx: float, cy: float,
                length: float, width: float,
                heading: float) -> np.ndarray:
    """Compute 4 corner points of an OBB.

    Returns:
        (4, 2) array of [x, y] corners, counter-clockwise from front-right.
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    # Half-extents along heading (forward) and perpendicular (left)
    hl = length / 2.0    # half-length
    hw = width / 2.0     # half-width

    # Forward and right unit vectors
    # heading=0 → forward along +x
    dx_fwd, dy_fwd = cos_h * hl, sin_h * hl
    dx_rgt, dy_rgt = sin_h * hw, -cos_h * hw

    return np.array([
        [cx + dx_fwd + dx_rgt, cy + dy_fwd + dy_rgt],  # front-right
        [cx + dx_fwd - dx_rgt, cy + dy_fwd - dy_rgt],  # front-left
        [cx - dx_fwd - dx_rgt, cy - dy_fwd - dy_rgt],  # rear-left
        [cx - dx_fwd + dx_rgt, cy - dy_fwd + dy_rgt],  # rear-right
    ], dtype=np.float64)


def _get_axes(corners: np.ndarray) -> np.ndarray:
    """Get the 2 unique edge normals (potential separating axes) of an OBB.

    For a rectangle, only 2 unique normals exist (edges 0-1 and 1-2).
    """
    edges = []
    for i in range(2):  # only need 2 edges for a rectangle
        edge = corners[(i + 1) % 4] - corners[i]
        # Normal (perpendicular)
        normal = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-12:
            normal = normal / norm_len
        edges.append(normal)
    return np.array(edges)


def _project(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project all corners onto an axis, return (min, max) scalar values."""
    dots = corners @ axis
    return float(dots.min()), float(dots.max())


def obb_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """Test if two OBBs overlap using the Separating Axis Theorem.

    Returns True if the OBBs overlap (collide).
    """
    axes_a = _get_axes(corners_a)
    axes_b = _get_axes(corners_b)

    for axis in np.vstack([axes_a, axes_b]):
        min_a, max_a = _project(corners_a, axis)
        min_b, max_b = _project(corners_b, axis)
        if max_a < min_b or max_b < min_a:
            return False  # separating axis found → no overlap
    return True


def obb_min_distance(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Approximate minimum distance between two OBBs (vectorized).

    Uses vectorized numpy operations for all point-point and
    point-segment distances. For overlapping OBBs returns 0.
    """
    if obb_overlap(corners_a, corners_b):
        return 0.0

    # Vectorized point-point distances: (4,1,2) - (1,4,2) → (4,4)
    diff = corners_a[:, None, :] - corners_b[None, :, :]  # (4,4,2)
    pp_dist = np.sqrt((diff ** 2).sum(axis=2))             # (4,4)
    min_dist = float(pp_dist.min())

    # Point-segment distances (vectorized per edge set)
    for c_main, c_other in [(corners_a, corners_b), (corners_b, corners_a)]:
        # 4 edges: c_main[j] → c_main[(j+1)%4]
        e0 = c_main                                         # (4,2)
        e1 = np.roll(c_main, -1, axis=0)                    # (4,2)
        ab = e1 - e0                                         # (4,2)
        ab_sq = (ab ** 2).sum(axis=1)                        # (4,)

        for p in c_other:                                    # iterate 4 pts
            ap = p - e0                                      # (4,2)
            dot_ap_ab = (ap * ab).sum(axis=1)                # (4,)
            # Safe division
            safe_sq = np.where(ab_sq > 1e-12, ab_sq, 1.0)
            t = np.clip(dot_ap_ab / safe_sq, 0.0, 1.0)
            t = np.where(ab_sq > 1e-12, t, 0.0)
            proj = e0 + t[:, None] * ab                      # (4,2)
            d = np.sqrt(((p - proj) ** 2).sum(axis=1))       # (4,)
            d_min = float(d.min())
            if d_min < min_dist:
                min_dist = d_min

    return min_dist


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to line segment a-b."""
    ab = b - a
    ap = p - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, float(np.dot(ap, ab)) / ab_sq))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


# ---------------------------------------------------------------------------
# TTC Computation
# ---------------------------------------------------------------------------

def compute_ttc_obb(
    ego: AgentBox,
    target: AgentBox,
    dt: float = DT_STEP,
    t_max: float = T_MAX_S,
) -> TTCResult:
    """Compute 2D OBB-based TTC between ego and target.

    Constant-velocity forward projection with fixed heading geometry.
    Optimized: diverging pairs skip forward projection.

    Args:
        ego: SDC agent state
        target: target agent state
        dt: projection time step (seconds)
        t_max: maximum projection horizon (seconds)

    Returns:
        TTCResult with all derived fields
    """
    if not ego.valid or not target.valid:
        return TTCResult(
            derived_ttc_2d_s=float("nan"),
            derived_dtc_m=float("nan"),
            derived_closing_speed_mps=float("nan"),
            derived_overlap_now=False,
            derived_hit_future=False,
            derived_pair_valid=False,
            derived_ttc_status="invalid",
        )

    # Validate geometry
    if (ego.length <= 0 or ego.width <= 0
            or target.length <= 0 or target.width <= 0):
        return TTCResult(
            derived_ttc_2d_s=float("nan"),
            derived_dtc_m=float("nan"),
            derived_closing_speed_mps=float("nan"),
            derived_overlap_now=False,
            derived_hit_future=False,
            derived_pair_valid=False,
            derived_ttc_status="invalid",
        )

    # Closing speed: rate of decrease of center-to-center distance
    dx = target.cx - ego.cx
    dy = target.cy - ego.cy
    range_now = math.sqrt(dx ** 2 + dy ** 2)

    if range_now > 1e-9:
        ux = dx / range_now
        uy = dy / range_now
        rel_vx = target.vx - ego.vx
        rel_vy = target.vy - ego.vy
        closing_speed = -(rel_vx * ux + rel_vy * uy)
    else:
        closing_speed = 0.0

    # Current corners
    corners_ego_0 = obb_corners(ego.cx, ego.cy, ego.length, ego.width, ego.heading)
    corners_tgt_0 = obb_corners(target.cx, target.cy, target.length, target.width, target.heading)

    # Check current overlap
    overlap_now = obb_overlap(corners_ego_0, corners_tgt_0)
    if overlap_now:
        return TTCResult(
            derived_ttc_2d_s=0.0,
            derived_dtc_m=0.0,
            derived_closing_speed_mps=closing_speed,
            derived_overlap_now=True,
            derived_hit_future=True,
            derived_pair_valid=True,
            derived_ttc_status="event",
        )

    # Early exit: if pair is diverging fast and far, skip forward projection
    # This avoids 100 expensive SAT checks for the vast majority of pairs
    half_diag = 0.5 * math.sqrt(
        max(ego.length, target.length) ** 2 + max(ego.width, target.width) ** 2
    )
    if closing_speed <= 0 and range_now > 2 * half_diag:
        # Diverging or parallel — will never collide under constant velocity
        return TTCResult(
            derived_ttc_2d_s=t_max,
            derived_dtc_m=max(0.0, range_now - 2 * half_diag),
            derived_closing_speed_mps=closing_speed,
            derived_overlap_now=False,
            derived_hit_future=False,
            derived_pair_valid=True,
            derived_ttc_status="right_censored",
        )

    # Additional check: even if closing, will they arrive within t_max?
    if closing_speed > 0 and range_now > 2 * half_diag:
        min_ttc_estimate = (range_now - 2 * half_diag) / closing_speed
        if min_ttc_estimate > t_max:
            return TTCResult(
                derived_ttc_2d_s=t_max,
                derived_dtc_m=max(0.0, range_now - 2 * half_diag),
                derived_closing_speed_mps=closing_speed,
                derived_overlap_now=False,
                derived_hit_future=False,
                derived_pair_valid=True,
                derived_ttc_status="right_censored",
            )

    # Forward projection — only for approaching pairs within possible range
    n_steps = int(math.ceil(t_max / dt))
    min_dist = range_now  # approximate

    # Precompute OBB template offsets (heading-fixed)
    cos_e, sin_e = math.cos(ego.heading), math.sin(ego.heading)
    hl_e, hw_e = ego.length / 2, ego.width / 2
    ego_offsets = np.array([
        [cos_e * hl_e + sin_e * hw_e, sin_e * hl_e - cos_e * hw_e],
        [cos_e * hl_e - sin_e * hw_e, sin_e * hl_e + cos_e * hw_e],
        [-cos_e * hl_e - sin_e * hw_e, -sin_e * hl_e + cos_e * hw_e],
        [-cos_e * hl_e + sin_e * hw_e, -sin_e * hl_e - cos_e * hw_e],
    ], dtype=np.float64)

    cos_t, sin_t = math.cos(target.heading), math.sin(target.heading)
    hl_t, hw_t = target.length / 2, target.width / 2
    tgt_offsets = np.array([
        [cos_t * hl_t + sin_t * hw_t, sin_t * hl_t - cos_t * hw_t],
        [cos_t * hl_t - sin_t * hw_t, sin_t * hl_t + cos_t * hw_t],
        [-cos_t * hl_t - sin_t * hw_t, -sin_t * hl_t + cos_t * hw_t],
        [-cos_t * hl_t + sin_t * hw_t, -sin_t * hl_t - cos_t * hw_t],
    ], dtype=np.float64)

    for step in range(1, n_steps + 1):
        t = step * dt

        # Fast center-to-center distance check first
        ego_cx_t = ego.cx + ego.vx * t
        ego_cy_t = ego.cy + ego.vy * t
        tgt_cx_t = target.cx + target.vx * t
        tgt_cy_t = target.cy + target.vy * t

        dist_ct = math.sqrt((ego_cx_t - tgt_cx_t) ** 2 + (ego_cy_t - tgt_cy_t) ** 2)
        if dist_ct > range_now + 5.0:
            # Getting further apart — can break early
            break
        if dist_ct < min_dist:
            min_dist = dist_ct

        # Only do SAT check if close enough
        if dist_ct > 2 * half_diag + 2.0:
            continue

        # Construct corners using precomputed offsets
        ego_center = np.array([ego_cx_t, ego_cy_t])
        tgt_center = np.array([tgt_cx_t, tgt_cy_t])
        corners_ego_t = ego_offsets + ego_center
        corners_tgt_t = tgt_offsets + tgt_center

        if obb_overlap(corners_ego_t, corners_tgt_t):
            return TTCResult(
                derived_ttc_2d_s=t,
                derived_dtc_m=max(0.0, min_dist - 2 * half_diag),
                derived_closing_speed_mps=closing_speed,
                derived_overlap_now=False,
                derived_hit_future=True,
                derived_pair_valid=True,
                derived_ttc_status="event",
            )

    # No collision within horizon → right-censored
    return TTCResult(
        derived_ttc_2d_s=t_max,
        derived_dtc_m=max(0.0, min_dist - 2 * half_diag),
        derived_closing_speed_mps=closing_speed,
        derived_overlap_now=False,
        derived_hit_future=False,
        derived_pair_valid=True,
        derived_ttc_status="right_censored",
    )
