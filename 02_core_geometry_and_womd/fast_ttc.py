#!/usr/bin/env python3
"""
Fast Circle-Based TTC Calculator
===================================
Analytical TTC using circular bounding approximation.
O(1) per pair via quadratic solve, ~100x faster than OBB SAT.

Each agent is modeled as a circumscribed circle with radius = 0.5 * sqrt(length² + width²).
TTC is the first time the two circles overlap under constant velocity.

Method ID: circle_circumscribed_cv_v1

Assumptions (same as OBB version):
  - Constant velocity extrapolation
  - 2D collision proxy (z ignored)
  - Current overlap → TTC = 0
  - No collision within horizon → right-censored at T_MAX_S
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration (same as obb_ttc)
# ---------------------------------------------------------------------------
T_MAX_S = 10.0
T_MIN_S = 0.05
DT_STEP = 0.1  # kept for API compat


@dataclass
class AgentBox:
    """Agent 2D state."""
    cx: float
    cy: float
    length: float
    width: float
    heading: float
    vx: float
    vy: float
    valid: bool = True

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def radius(self) -> float:
        """Bounding circle radius = half of diagonal."""
        return 0.5 * math.sqrt(self.length ** 2 + self.width ** 2)


@dataclass
class TTCResult:
    """TTC computation result."""
    derived_ttc_2d_s: float
    derived_dtc_m: float
    derived_closing_speed_mps: float
    derived_overlap_now: bool
    derived_hit_future: bool
    derived_pair_valid: bool
    derived_ttc_status: str


def _invalid_result():
    return TTCResult(
        derived_ttc_2d_s=float("nan"),
        derived_dtc_m=float("nan"),
        derived_closing_speed_mps=float("nan"),
        derived_overlap_now=False,
        derived_hit_future=False,
        derived_pair_valid=False,
        derived_ttc_status="invalid",
    )


def compute_ttc_fast(
    ego: AgentBox,
    target: AgentBox,
    t_max: float = T_MAX_S,
) -> TTCResult:
    """Analytical circle-based TTC.

    Solves: |p(t)|^2 = R^2  where p(t) = p0 + v * t
    This is a quadratic: |v|^2 * t^2 + 2*(p0·v)*t + |p0|^2 - R^2 = 0

    Args:
        ego: SDC agent state
        target: target agent state
        t_max: maximum horizon (seconds)

    Returns:
        TTCResult
    """
    if not ego.valid or not target.valid:
        return _invalid_result()
    if ego.length <= 0 or ego.width <= 0 or target.length <= 0 or target.width <= 0:
        return _invalid_result()

    # Combined bounding radius
    R = ego.radius + target.radius

    # Relative position (target relative to ego)
    dx = target.cx - ego.cx
    dy = target.cy - ego.cy
    range_now = math.sqrt(dx * dx + dy * dy)

    # Relative velocity
    dvx = target.vx - ego.vx
    dvy = target.vy - ego.vy

    # Closing speed (positive = approaching)
    if range_now > 1e-9:
        ux = dx / range_now
        uy = dy / range_now
        closing_speed = -(dvx * ux + dvy * uy)
    else:
        closing_speed = 0.0

    # Current overlap check
    if range_now <= R:
        return TTCResult(
            derived_ttc_2d_s=0.0,
            derived_dtc_m=0.0,
            derived_closing_speed_mps=closing_speed,
            derived_overlap_now=True,
            derived_hit_future=True,
            derived_pair_valid=True,
            derived_ttc_status="event",
        )

    # Quadratic solve: a*t^2 + b*t + c = 0
    # where dist(t)^2 = |p0 + v*t|^2
    # p0 = (dx, dy), v = (dvx, dvy)
    a = dvx * dvx + dvy * dvy             # |v|^2
    b = 2.0 * (dx * dvx + dy * dvy)       # 2 * (p0 · v)
    c = dx * dx + dy * dy - R * R          # |p0|^2 - R^2

    # Distance to closest approach
    # t_closest = -b / (2a) — time of closest approach
    if a > 1e-12:
        t_closest = -b / (2.0 * a)
        if 0 < t_closest <= t_max:
            dist_closest = math.sqrt(max(0.0,
                a * t_closest * t_closest + b * t_closest + c + R * R)) - R
            min_dist = max(0.0, dist_closest)
        else:
            min_dist = max(0.0, range_now - R)
    else:
        # No relative motion
        min_dist = max(0.0, range_now - R)

    # Solve quadratic for collision time
    discriminant = b * b - 4.0 * a * c

    if a < 1e-12:
        # No relative velocity → never collide (already checked overlap)
        return TTCResult(
            derived_ttc_2d_s=t_max,
            derived_dtc_m=min_dist,
            derived_closing_speed_mps=closing_speed,
            derived_overlap_now=False,
            derived_hit_future=False,
            derived_pair_valid=True,
            derived_ttc_status="right_censored",
        )

    if discriminant < 0:
        # Never reach contact distance → right-censored
        return TTCResult(
            derived_ttc_2d_s=t_max,
            derived_dtc_m=min_dist,
            derived_closing_speed_mps=closing_speed,
            derived_overlap_now=False,
            derived_hit_future=False,
            derived_pair_valid=True,
            derived_ttc_status="right_censored",
        )

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # First positive root within horizon
    ttc = None
    if 0 < t1 <= t_max:
        ttc = t1
    elif 0 < t2 <= t_max:
        ttc = t2

    if ttc is not None:
        return TTCResult(
            derived_ttc_2d_s=ttc,
            derived_dtc_m=0.0,  # will reach contact
            derived_closing_speed_mps=closing_speed,
            derived_overlap_now=False,
            derived_hit_future=True,
            derived_pair_valid=True,
            derived_ttc_status="event",
        )

    # No collision within horizon
    return TTCResult(
        derived_ttc_2d_s=t_max,
        derived_dtc_m=min_dist,
        derived_closing_speed_mps=closing_speed,
        derived_overlap_now=False,
        derived_hit_future=False,
        derived_pair_valid=True,
        derived_ttc_status="right_censored",
    )


def compute_ttc_obb(*args, **kwargs):
    """DEPRECATED: This alias is misleading. Use compute_ttc_fast() for Circle TTC
    or compute_ttc_obb_swept() from obb_ttc_swept.py for actual OBB TTC."""
    raise DeprecationWarning(
        "compute_ttc_obb is a misleading alias for Circle TTC. "
        "Use compute_ttc_fast() (circle_circumscribed_cv_v1) or "
        "compute_ttc_obb_swept() (obb_swept_sat_cv_fixed_heading_v1)."
    )
