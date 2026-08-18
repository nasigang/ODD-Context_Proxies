#!/usr/bin/env python3
"""
Continuous Swept-OBB TTC (Analytical SAT)
============================================
Method ID: obb_swept_sat_cv_fixed_heading_v1

Computes exact first contact time between two 2D Oriented Bounding Boxes
under constant translational velocity and fixed heading.

Algorithm:
  1. Compute 4 separating axes from the two OBB edge normals.
  2. For each axis â:
     - d(t) = (c₂ - c₁ + (v₂ - v₁)·t) · â  — projected center distance
     - r_sum = |half₁ · â| + |half₂ · â|      — sum of support radii
     - Overlap iff |d(t)| ≤ r_sum
     - Solve for interval [t_enter, t_exit]
  3. OBB contact interval = intersection of all 4 intervals ∩ [0, horizon]
  4. entry ≤ exit → TTC = max(0, entry); else right-censored

Assumptions:
  - Constant velocity (no acceleration)
  - Fixed heading (no yaw rate) → OBB orientation constant
  - Fixed dimensions (length/width constant during projection)
  - 2D only (z ignored)
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T_MAX_S = 10.0
T_MIN_S = 0.05
METHOD_ID = "obb_swept_sat_cv_fixed_heading_v1"
EPS = 1e-12
CONTACT_TOL = 1e-9  # tolerance for tangent contact


@dataclass
class OBBAgent:
    """2D agent state for OBB TTC."""
    cx: float
    cy: float
    length: float
    width: float
    heading: float  # radians
    vx: float
    vy: float
    valid: bool = True

    @property
    def half_extents(self) -> Tuple[float, float]:
        """Half-length along local x, half-width along local y."""
        return self.length / 2.0, self.width / 2.0

    @property
    def axes(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Two unit edge-normal axes from heading."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (c, s), (-s, c)

    @property
    def circle_radius(self) -> float:
        """Circumscribed circle radius for invariant checks."""
        return 0.5 * math.sqrt(self.length ** 2 + self.width ** 2)

    def get_corners(self) -> np.ndarray:
        """Return 4x2 array of 2D corner coordinates in counter-clockwise order."""
        hl, hw = self.length / 2.0, self.width / 2.0
        c, s = math.cos(self.heading), math.sin(self.heading)
        ux, uy = c, s
        vx, vy = -s, c
        return np.array([
            [self.cx + hl * ux + hw * vx, self.cy + hl * uy + hw * vy],
            [self.cx - hl * ux + hw * vx, self.cy - hl * uy + hw * vy],
            [self.cx - hl * ux - hw * vx, self.cy - hl * uy - hw * vy],
            [self.cx + hl * ux - hw * vx, self.cy + hl * uy - hw * vy],
        ], dtype=np.float64)


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance from point (px, py) to line segment (ax, ay)-(bx, by)."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.sqrt((px - qx) ** 2 + (py - qy) ** 2)


def compute_obb_boundary_clearance(ego: OBBAgent, target: OBBAgent) -> float:
    """
    Compute exact Euclidean boundary clearance between two 2D OBB polygons.
    Returns 0.0 if overlapping or touching, else minimum boundary distance in meters.
    """
    if not ego.valid or not target.valid:
        return float("nan")

    # Fast SAT overlap check
    e_hl, e_hw = ego.half_extents
    t_hl, t_hw = target.half_extents
    e_ax, e_ay = ego.axes
    t_ax, t_ay = target.axes
    dp = (target.cx - ego.cx, target.cy - ego.cy)

    overlap = True
    for axis in [e_ax, e_ay, t_ax, t_ay]:
        d_proj = abs(_dot(dp, axis))
        r_ego = _support_radius(e_hl, e_hw, e_ax, e_ay, axis)
        r_tgt = _support_radius(t_hl, t_hw, t_ax, t_ay, axis)
        if d_proj > r_ego + r_tgt + CONTACT_TOL:
            overlap = False
            break

    if overlap:
        return 0.0

    # Compute minimum point-to-segment distance across all 4x4 polygon edges
    c_ego = ego.get_corners()
    c_tgt = target.get_corners()

    min_dist = float("inf")
    # Ego vertices to target edges
    for i in range(4):
        px, py = c_ego[i, 0], c_ego[i, 1]
        for j in range(4):
            ax, ay = c_tgt[j, 0], c_tgt[j, 1]
            bx, by = c_tgt[(j + 1) % 4, 0], c_tgt[(j + 1) % 4, 1]
            d = _point_to_segment_distance(px, py, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d

    # Target vertices to ego edges
    for i in range(4):
        px, py = c_tgt[i, 0], c_tgt[i, 1]
        for j in range(4):
            ax, ay = c_ego[j, 0], c_ego[j, 1]
            bx, by = c_ego[(j + 1) % 4, 0], c_ego[(j + 1) % 4, 1]
            d = _point_to_segment_distance(px, py, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d

    return float(min_dist)


@dataclass
class SweptTTCResult:
    """Result of swept-OBB TTC computation."""
    ttc_s: float
    contact_entry: float   # first contact time (may be 0)
    contact_exit: float    # last contact time
    overlap_now: bool
    hit_future: bool
    pair_valid: bool
    status: str            # 6-state taxonomy
    method_id: str = METHOD_ID


def _invalid_result() -> SweptTTCResult:
    return SweptTTCResult(
        ttc_s=float("nan"), contact_entry=float("nan"),
        contact_exit=float("nan"), overlap_now=False,
        hit_future=False, pair_valid=False, status="invalid"
    )


def _axis_interval(
    d0: float, dv: float, r_sum: float, t_max: float
) -> Tuple[float, float]:
    """
    Compute the time interval [t_enter, t_exit] where |d0 + dv*t| <= r_sum.

    d(t) = d0 + dv * t
    |d(t)| <= r_sum  ⟺  -r_sum <= d0 + dv*t <= r_sum

    Returns (t_enter, t_exit). If no overlap possible, returns (inf, -inf).
    """
    if abs(dv) < EPS:
        # Static on this axis — either always overlapping or never
        if abs(d0) <= r_sum + CONTACT_TOL:
            return 0.0, t_max
        else:
            return float("inf"), float("-inf")

    # Solve d0 + dv*t = ±r_sum
    t_a = (-r_sum - d0) / dv
    t_b = (r_sum - d0) / dv

    t_enter = min(t_a, t_b)
    t_exit = max(t_a, t_b)

    return t_enter, t_exit


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _support_radius(
    half_l: float, half_w: float,
    ax_local: Tuple[float, float],  # local x-axis
    ay_local: Tuple[float, float],  # local y-axis
    axis: Tuple[float, float]       # separating axis
) -> float:
    """Project OBB half-extents onto a separating axis → support radius."""
    return (abs(_dot(ax_local, axis)) * half_l +
            abs(_dot(ay_local, axis)) * half_w)


def compute_ttc_obb_swept(
    ego: OBBAgent,
    target: OBBAgent,
    t_max: float = T_MAX_S,
) -> SweptTTCResult:
    """Analytical swept-SAT TTC between two OBBs.

    Constant velocity, fixed heading.

    Args:
        ego: SDC agent state
        target: target agent state
        t_max: maximum projection horizon (seconds)

    Returns:
        SweptTTCResult with first contact time
    """
    if not ego.valid or not target.valid:
        return _invalid_result()
    if ego.length <= 0 or ego.width <= 0 or target.length <= 0 or target.width <= 0:
        return _invalid_result()

    # Half-extents
    e_hl, e_hw = ego.half_extents
    t_hl, t_hw = target.half_extents

    # Local axes
    e_ax, e_ay = ego.axes
    t_ax, t_ay = target.axes

    # Relative position and velocity
    dx = target.cx - ego.cx
    dy = target.cy - ego.cy
    dvx = target.vx - ego.vx
    dvy = target.vy - ego.vy
    dp = (dx, dy)
    dv = (dvx, dvy)

    # 4 separating axes: 2 from ego, 2 from target
    all_axes = [e_ax, e_ay, t_ax, t_ay]

    # Global overlap interval = intersection of all axis intervals with [0, t_max]
    global_enter = 0.0
    global_exit = t_max

    for axis in all_axes:
        # Project center distance onto axis
        d0 = _dot(dp, axis)
        dv_proj = _dot(dv, axis)

        # Support radii
        r_ego = _support_radius(e_hl, e_hw, e_ax, e_ay, axis)
        r_tgt = _support_radius(t_hl, t_hw, t_ax, t_ay, axis)
        r_sum = r_ego + r_tgt

        t_enter, t_exit = _axis_interval(d0, dv_proj, r_sum, t_max)

        global_enter = max(global_enter, t_enter)
        global_exit = min(global_exit, t_exit)

        if global_enter > global_exit + CONTACT_TOL:
            # No overlap possible — separated on this axis
            return SweptTTCResult(
                ttc_s=t_max, contact_entry=float("nan"),
                contact_exit=float("nan"), overlap_now=False,
                hit_future=False, pair_valid=True,
                status="right_censored"
            )

    # Check if we have a valid contact interval
    if global_enter > global_exit + CONTACT_TOL:
        return SweptTTCResult(
            ttc_s=t_max, contact_entry=float("nan"),
            contact_exit=float("nan"), overlap_now=False,
            hit_future=False, pair_valid=True,
            status="right_censored"
        )

    # Clamp to [0, t_max]
    entry = max(0.0, global_enter)
    exit_t = min(t_max, global_exit)

    if entry > exit_t + CONTACT_TOL:
        return SweptTTCResult(
            ttc_s=t_max, contact_entry=float("nan"),
            contact_exit=float("nan"), overlap_now=False,
            hit_future=False, pair_valid=True,
            status="right_censored"
        )

    overlap_now = (entry <= CONTACT_TOL)  # contact at t ≈ 0

    if overlap_now:
        status = "current_geometry_overlap"
    else:
        status = "future_contact_event"

    return SweptTTCResult(
        ttc_s=entry,
        contact_entry=entry,
        contact_exit=exit_t,
        overlap_now=overlap_now,
        hit_future=True,
        pair_valid=True,
        status=status
    )


def compute_ttc_obb_dense_sat(
    ego: OBBAgent,
    target: OBBAgent,
    t_max: float = T_MAX_S,
    dt: float = 0.001,
) -> SweptTTCResult:
    """Dense step-wise SAT for testing only.

    Steps through time at dt intervals, checks OBB overlap at each step.
    This is O(t_max/dt) per pair — test reference only.
    """
    if not ego.valid or not target.valid:
        return _invalid_result()
    if ego.length <= 0 or ego.width <= 0 or target.length <= 0 or target.width <= 0:
        return _invalid_result()

    e_hl, e_hw = ego.half_extents
    t_hl, t_hw = target.half_extents
    e_ax, e_ay = ego.axes
    t_ax, t_ay = target.axes

    dvx = target.vx - ego.vx
    dvy = target.vy - ego.vy

    n_steps = int(math.ceil(t_max / dt)) + 1
    first_contact = None

    for i in range(n_steps):
        t = i * dt
        if t > t_max:
            break

        # Positions at time t
        dx = (target.cx + target.vx * t) - (ego.cx + ego.vx * t)
        dy = (target.cy + target.vy * t) - (ego.cy + ego.vy * t)
        dp = (dx, dy)

        # Check all 4 axes
        separated = False
        for axis in [e_ax, e_ay, t_ax, t_ay]:
            d_proj = abs(_dot(dp, axis))
            r_ego = _support_radius(e_hl, e_hw, e_ax, e_ay, axis)
            r_tgt = _support_radius(t_hl, t_hw, t_ax, t_ay, axis)
            if d_proj > r_ego + r_tgt + CONTACT_TOL:
                separated = True
                break

        if not separated:
            first_contact = t
            break

    if first_contact is not None:
        overlap_now = first_contact <= dt + CONTACT_TOL
        status = "current_geometry_overlap" if overlap_now else "future_contact_event"
        return SweptTTCResult(
            ttc_s=first_contact,
            contact_entry=first_contact, contact_exit=first_contact,
            overlap_now=overlap_now, hit_future=True, pair_valid=True,
            status=status
        )

    return SweptTTCResult(
        ttc_s=t_max, contact_entry=float("nan"),
        contact_exit=float("nan"), overlap_now=False,
        hit_future=False, pair_valid=True, status="right_censored"
    )
