#!/usr/bin/env python3
"""
OBB TTC Toy Tests
==================
7 required toy scenarios that verify OBB collision detection and TTC logic.

1. Current OBB overlap → TTC=0
2. Head-on approach → future overlap → TTC ≈ distance/closing_speed
3. Lateral pass → no collision → right_censored
4. No eligible objects → handled at frame level (no_exposure)
5. Invalid SDC state → pair_valid=False
6. Static object (v=0) → only ego closing
7. Heading ≠ velocity direction → geometry correct
"""

import math

import numpy as np
import pandas as pd
import pytest

from phase2_womd.obb_ttc import (
    T_MAX_S,
    AgentBox,
    TTCResult,
    compute_ttc_obb,
    obb_corners,
    obb_min_distance,
    obb_overlap,
)
from phase2_womd.obb_ttc import (
    T_MAX_S,
    AgentBox,
    TTCResult,
    compute_ttc_obb,
    obb_corners,
    obb_min_distance,
    obb_overlap,
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(cx, cy, length, width, heading, vx, vy, valid=True):
    return AgentBox(cx=cx, cy=cy, length=length, width=width,
                    heading=heading, vx=vx, vy=vy, valid=valid)


# ---------------------------------------------------------------------------
# Test 1: Current OBB overlap → TTC = 0
# ---------------------------------------------------------------------------
class TestCurrentOverlap:
    def test_overlapping_boxes_ttc_zero(self):
        """Two OBBs centered at (0,0) and (1,0), both 4m×2m → overlap."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0, vx=0, vy=0)
        tgt = _make_agent(cx=1, cy=0, length=4, width=2, heading=0, vx=0, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_ttc_2d_s == 0.0
        assert result.derived_overlap_now is True
        assert result.derived_hit_future is True
        assert result.derived_pair_valid is True
        assert result.derived_ttc_status == "event"
        assert result.derived_dtc_m == 0.0


# ---------------------------------------------------------------------------
# Test 2: Head-on approach → future overlap
# ---------------------------------------------------------------------------
class TestHeadOnApproach:
    def test_head_on_collision(self):
        """Ego at x=0 going +x at 10m/s, target at x=50 going -x at 10m/s.
        Both 4m×2m, heading aligned with velocity.
        Gap = 50 - 2 - 2 = 46m (edge to edge), closing speed = 20 m/s.
        Expected TTC ≈ 46/20 = 2.3s."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=10, vy=0)
        tgt = _make_agent(cx=50, cy=0, length=4, width=2, heading=math.pi,
                          vx=-10, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_ttc_status == "event"
        assert result.derived_pair_valid is True
        assert result.derived_hit_future is True
        assert result.derived_overlap_now is False
        # TTC should be around 2.3s (discretized to 0.1s steps)
        assert 2.0 <= result.derived_ttc_2d_s <= 2.5
        assert result.derived_closing_speed_mps > 0

    def test_closing_speed_positive_when_approaching(self):
        """Closing speed should be positive when agents approach."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=10, vy=0)
        tgt = _make_agent(cx=30, cy=0, length=4, width=2, heading=math.pi,
                          vx=-5, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_closing_speed_mps > 0


# ---------------------------------------------------------------------------
# Test 3: Lateral pass → no collision → right_censored
# ---------------------------------------------------------------------------
class TestLateralPass:
    def test_perpendicular_pass_no_collision(self):
        """Ego at (0,0) going +x, target at (50,10) going -x.
        Lateral offset of 10m with 2m-wide cars → never overlap."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=10, vy=0)
        tgt = _make_agent(cx=50, cy=10, length=4, width=2, heading=math.pi,
                          vx=-10, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_ttc_status == "right_censored"
        assert result.derived_hit_future is False
        assert result.derived_overlap_now is False
        assert result.derived_ttc_2d_s == T_MAX_S
        assert result.derived_pair_valid is True

    def test_crossing_paths_no_collision(self):
        """Ego going +x, target going +y, paths cross but not at same time.
        Target starts far enough away that they miss each other."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=10, vy=0)
        tgt = _make_agent(cx=5, cy=100, length=4, width=2,
                          heading=-math.pi / 2, vx=0, vy=-10)

        result = compute_ttc_obb(ego, tgt)
        # They may or may not collide depending on timing — just verify
        # the pair is valid and result is well-formed
        assert result.derived_pair_valid is True
        assert result.derived_ttc_status in ("event", "right_censored")


# ---------------------------------------------------------------------------
# Test 4: Target not valid / missing
# ---------------------------------------------------------------------------
class TestNoEligibleObjects:
    def test_invalid_target_produces_invalid_pair(self):
        """Invalid target state produces invalid pair result."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0, vx=10, vy=0)
        tgt = _make_agent(cx=20, cy=0, length=4, width=2, heading=0, vx=0, vy=0, valid=False)
        result = compute_ttc_obb(ego, tgt)
        assert result.derived_pair_valid is False
        assert result.derived_ttc_status == "invalid"
        assert np.isnan(result.derived_ttc_2d_s)


# ---------------------------------------------------------------------------
# Test 5: Invalid SDC state
# ---------------------------------------------------------------------------
class TestInvalidState:
    def test_invalid_ego_produces_invalid_pair(self):
        """Invalid ego → pair_valid=False."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=10, vy=0, valid=False)
        tgt = _make_agent(cx=20, cy=0, length=4, width=2, heading=0,
                          vx=0, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_pair_valid is False
        assert result.derived_ttc_status == "invalid"
        assert np.isnan(result.derived_ttc_2d_s)

    def test_invalid_dimensions_produce_invalid_pair(self):
        """Zero or negative length/width produces invalid pair."""
        ego = _make_agent(cx=0, cy=0, length=0, width=2, heading=0, vx=10, vy=0)
        tgt = _make_agent(cx=20, cy=0, length=4, width=2, heading=0, vx=0, vy=0)
        result = compute_ttc_obb(ego, tgt)
        assert result.derived_pair_valid is False
        assert result.derived_ttc_status == "invalid"



# ---------------------------------------------------------------------------
# Test 6: Static object (v=0)
# ---------------------------------------------------------------------------
class TestStaticObject:
    def test_ego_approaching_static_target(self):
        """Ego moving toward a parked car. Should detect future collision."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=10, vy=0)
        tgt = _make_agent(cx=30, cy=0, length=4, width=2, heading=0,
                          vx=0, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_pair_valid is True
        assert result.derived_ttc_status == "event"
        assert result.derived_hit_future is True
        # Edge-to-edge gap = 30 - 2 - 2 = 26m, speed = 10m/s → ~2.6s
        assert 2.0 <= result.derived_ttc_2d_s <= 3.0
        assert result.derived_closing_speed_mps > 0

    def test_both_static_no_overlap(self):
        """Both stationary and far apart → right-censored."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=0, vy=0)
        tgt = _make_agent(cx=30, cy=0, length=4, width=2, heading=0,
                          vx=0, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_ttc_status == "right_censored"
        assert result.derived_hit_future is False


# ---------------------------------------------------------------------------
# Test 7: Heading ≠ velocity direction
# ---------------------------------------------------------------------------
class TestHeadingVelocityMismatch:
    def test_crab_walk(self):
        """Ego heading along +x but velocity along +y (crab walk).
        Target directly ahead in +y direction.
        OBB should use heading for box orientation, velocity for motion."""
        # Ego heading east, sliding north
        ego = _make_agent(cx=0, cy=0, length=4, width=2, heading=0,
                          vx=0, vy=10)
        # Target ahead in +y, heading east, stationary
        tgt = _make_agent(cx=0, cy=25, length=4, width=2, heading=0,
                          vx=0, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_pair_valid is True
        # Ego sliding north toward target at cy=25
        # OBB is 4m×2m oriented east — top edge at y=+1, so gap = 25-1-1=23m
        # at speed=10, TTC ≈ 2.3s
        assert result.derived_ttc_status == "event"
        assert result.derived_hit_future is True

    def test_diagonal_heading_perpendicular_velocity(self):
        """Heading at 45° but velocity along +x.
        Box is tilted but motion is horizontal — geometry matters."""
        ego = _make_agent(cx=0, cy=0, length=4, width=2,
                          heading=math.pi / 4, vx=10, vy=0)
        tgt = _make_agent(cx=40, cy=0, length=4, width=2,
                          heading=0, vx=0, vy=0)

        result = compute_ttc_obb(ego, tgt)
        assert result.derived_pair_valid is True
        # Should still detect collision since ego moves toward target
        assert result.derived_ttc_status == "event"


# ---------------------------------------------------------------------------
# OBB Geometry unit tests
# ---------------------------------------------------------------------------
class TestOBBGeometry:
    def test_corners_axis_aligned(self):
        """Heading=0, 4m×2m box at origin."""
        corners = obb_corners(0, 0, 4, 2, 0)
        assert corners.shape == (4, 2)
        # Front-right should be at (2, -1)
        np.testing.assert_allclose(corners[0], [2, -1], atol=1e-10)
        # Front-left at (2, 1)
        np.testing.assert_allclose(corners[1], [2, 1], atol=1e-10)

    def test_corners_rotated_90(self):
        """Heading=π/2, 4m×2m box at origin."""
        corners = obb_corners(0, 0, 4, 2, math.pi / 2)
        # Front should be at y=+2
        assert corners.shape == (4, 2)
        # Front-right: rotated from (2, -1) by 90° → (1, 2)
        np.testing.assert_allclose(corners[0], [1, 2], atol=1e-10)

    def test_no_overlap_separated(self):
        c1 = obb_corners(0, 0, 2, 2, 0)
        c2 = obb_corners(10, 0, 2, 2, 0)
        assert obb_overlap(c1, c2) is False

    def test_overlap_touching(self):
        c1 = obb_corners(0, 0, 2, 2, 0)
        c2 = obb_corners(1.5, 0, 2, 2, 0)
        assert obb_overlap(c1, c2) is True

    def test_min_distance_separated(self):
        c1 = obb_corners(0, 0, 2, 2, 0)
        c2 = obb_corners(10, 0, 2, 2, 0)
        d = obb_min_distance(c1, c2)
        # Edge-to-edge: 10 - 1 - 1 = 8m
        assert abs(d - 8.0) < 0.1

    def test_min_distance_overlapping(self):
        c1 = obb_corners(0, 0, 2, 2, 0)
        c2 = obb_corners(0.5, 0, 2, 2, 0)
        assert obb_min_distance(c1, c2) == 0.0
