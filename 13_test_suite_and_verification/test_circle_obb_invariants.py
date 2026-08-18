#!/usr/bin/env python3
"""
Test Circle-OBB Containment Invariants.

The circumscribed circle always contains the OBB. Therefore:
  1. If OBB contact at t_obb → Circle must have contact at t_circle ≤ t_obb + tol
  2. Circle current_overlap → OBB may or may not overlap (Circle is larger)
  3. OBB current_overlap → Circle MUST overlap
  4. If both finite: Circle_TTC ≤ OBB_TTC + tolerance
"""
import math
import pytest
import numpy as np

from phase2_womd.fast_ttc import AgentBox, compute_ttc_fast
from phase2_womd.obb_ttc_swept import OBBAgent, compute_ttc_obb_swept

TOL = 0.05  # 50ms tolerance


def _circle_agent(cx, cy, l, w, h, vx, vy):
    return AgentBox(cx=cx, cy=cy, length=l, width=w, heading=h, vx=vx, vy=vy)


def _obb_agent(cx, cy, l, w, h, vx, vy):
    return OBBAgent(cx=cx, cy=cy, length=l, width=w, heading=h, vx=vx, vy=vy)


def _make_pair(cx1, cy1, l1, w1, h1, vx1, vy1,
               cx2, cy2, l2, w2, h2, vx2, vy2):
    c_ego = _circle_agent(cx1, cy1, l1, w1, h1, vx1, vy1)
    c_tgt = _circle_agent(cx2, cy2, l2, w2, h2, vx2, vy2)
    o_ego = _obb_agent(cx1, cy1, l1, w1, h1, vx1, vy1)
    o_tgt = _obb_agent(cx2, cy2, l2, w2, h2, vx2, vy2)
    return c_ego, c_tgt, o_ego, o_tgt


class TestContainmentInvariant:
    """OBB contact → Circle contact."""

    @pytest.mark.parametrize("seed", range(100))
    def test_random_containment(self, seed):
        """If OBB finds contact, Circle MUST also find contact."""
        rng = np.random.RandomState(seed)

        args = [
            0, 0,  # ego pos
            rng.uniform(3, 6), rng.uniform(1.5, 2.5),  # ego dims
            rng.uniform(-math.pi, math.pi),  # ego heading
            rng.uniform(-15, 15), rng.uniform(-15, 15),  # ego vel
            rng.uniform(-40, 40), rng.uniform(-40, 40),  # tgt pos
            rng.uniform(0.5, 6), rng.uniform(0.3, 2.5),  # tgt dims
            rng.uniform(-math.pi, math.pi),  # tgt heading
            rng.uniform(-15, 15), rng.uniform(-15, 15),  # tgt vel
        ]
        c_ego, c_tgt, o_ego, o_tgt = _make_pair(*args)

        circle_r = compute_ttc_fast(c_ego, c_tgt)
        obb_r = compute_ttc_obb_swept(o_ego, o_tgt)

        if obb_r.hit_future:
            # INVARIANT: Circle must also have contact
            assert circle_r.derived_hit_future or circle_r.derived_overlap_now, (
                f"Seed {seed}: OBB contact at {obb_r.ttc_s:.4f}s "
                f"but Circle is {circle_r.derived_ttc_status}"
            )

    @pytest.mark.parametrize("seed", range(100))
    def test_circle_ttc_leq_obb(self, seed):
        """If both have finite TTC, Circle TTC ≤ OBB TTC + tolerance."""
        rng = np.random.RandomState(seed)

        args = [
            0, 0,
            rng.uniform(3, 6), rng.uniform(1.5, 2.5),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-15, 15), rng.uniform(-15, 15),
            rng.uniform(-30, 30), rng.uniform(-30, 30),
            rng.uniform(0.5, 6), rng.uniform(0.3, 2.5),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-15, 15), rng.uniform(-15, 15),
        ]
        c_ego, c_tgt, o_ego, o_tgt = _make_pair(*args)

        circle_r = compute_ttc_fast(c_ego, c_tgt)
        obb_r = compute_ttc_obb_swept(o_ego, o_tgt)

        if obb_r.hit_future and circle_r.derived_hit_future:
            assert circle_r.derived_ttc_2d_s <= obb_r.ttc_s + TOL, (
                f"Seed {seed}: Circle TTC={circle_r.derived_ttc_2d_s:.4f} > "
                f"OBB TTC={obb_r.ttc_s:.4f} + tol={TOL}"
            )


class TestOBBOverlapImpliesCircleOverlap:
    """OBB current overlap → Circle must overlap."""

    def test_overlapping_boxes(self):
        c_ego, c_tgt, o_ego, o_tgt = _make_pair(
            0, 0, 4, 2, 0, 0, 0,
            1, 0, 4, 2, 0, 0, 0
        )
        obb_r = compute_ttc_obb_swept(o_ego, o_tgt)
        circle_r = compute_ttc_fast(c_ego, c_tgt)

        if obb_r.overlap_now:
            assert circle_r.derived_overlap_now, (
                "OBB overlap at t=0 but Circle does not overlap"
            )

    def test_rotated_overlap(self):
        c_ego, c_tgt, o_ego, o_tgt = _make_pair(
            0, 0, 4, 2, math.pi / 4, 0, 0,
            2, 2, 4, 2, -math.pi / 4, 0, 0
        )
        obb_r = compute_ttc_obb_swept(o_ego, o_tgt)
        circle_r = compute_ttc_fast(c_ego, c_tgt)

        if obb_r.overlap_now:
            assert circle_r.derived_overlap_now


class TestCircleFalsePositive:
    """Circle may detect contact that OBB does not (false positive) — this is expected."""

    def test_near_miss_corner(self):
        """Circle contact but OBB miss: corner case where circle extends beyond OBB."""
        # Diagonal of 4×2 box: sqrt(20)≈4.47, circle radius≈2.24
        # Two boxes side by side at offset that's within circle but outside OBB
        c_ego, c_tgt, o_ego, o_tgt = _make_pair(
            0, 0, 4, 2, 0, 0, 0,
            0, 4, 4, 2, 0, 0, -2  # approaching from above
        )
        circle_r = compute_ttc_fast(c_ego, c_tgt)
        obb_r = compute_ttc_obb_swept(o_ego, o_tgt)

        # This is allowed: Circle can have contact when OBB doesn't
        # (Circle is conservative = captures more)
        if circle_r.derived_hit_future and not obb_r.hit_future:
            pass  # Expected behavior — circle is more conservative
