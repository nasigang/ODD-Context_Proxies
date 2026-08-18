#!/usr/bin/env python3
"""
Test Circle TTC Contract — method_id, states, provenance, formula.
"""
import math
import pytest
from phase2_womd.fast_ttc import AgentBox, compute_ttc_fast, TTCResult, compute_ttc_obb


METHOD_ID = "circle_circumscribed_cv_v1"


def _box(cx, cy, l, w, h, vx, vy):
    return AgentBox(cx=cx, cy=cy, length=l, width=w, heading=h, vx=vx, vy=vy)


class TestCircleContractRadius:
    """Verify radius = 0.5 * sqrt(L² + W²) (half-diagonal, NOT max/2)."""

    def test_square(self):
        b = _box(0, 0, 4, 4, 0, 0, 0)
        expected = 0.5 * math.sqrt(16 + 16)  # = 2√2 ≈ 2.828
        assert abs(b.radius - expected) < 1e-10

    def test_rectangle(self):
        b = _box(0, 0, 4, 2, 0, 0, 0)
        expected = 0.5 * math.sqrt(16 + 4)  # = √5 ≈ 2.236
        assert abs(b.radius - expected) < 1e-10

    def test_not_max_half(self):
        """Radius must NOT be max(L,W)/2."""
        b = _box(0, 0, 4, 2, 0, 0, 0)
        wrong_radius = max(4, 2) / 2.0  # = 2.0
        assert abs(b.radius - wrong_radius) > 0.1, (
            "Radius should be half-diagonal, not max(L,W)/2"
        )


class TestCircleContractStates:
    """6-state taxonomy verification."""

    def test_invalid_pair(self):
        ego = _box(0, 0, 4, 2, 0, 10, 0)
        ego.valid = False
        tgt = _box(10, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_fast(ego, tgt)
        assert not r.derived_pair_valid
        assert r.derived_ttc_status == "invalid"

    def test_current_overlap(self):
        ego = _box(0, 0, 4, 2, 0, 0, 0)
        tgt = _box(1, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_fast(ego, tgt)
        assert r.derived_overlap_now
        assert r.derived_ttc_2d_s == 0.0
        # Status should indicate overlap
        assert r.derived_ttc_status == "event"

    def test_right_censored(self):
        ego = _box(0, 0, 4, 2, 0, 0, 0)
        tgt = _box(100, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_fast(ego, tgt)
        assert not r.derived_hit_future
        assert r.derived_ttc_status == "right_censored"

    def test_future_contact_event(self):
        ego = _box(0, 0, 4, 2, 0, 10, 0)
        tgt = _box(20, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_fast(ego, tgt)
        assert r.derived_hit_future
        assert not r.derived_overlap_now
        assert r.derived_ttc_status == "event"
        assert 0 < r.derived_ttc_2d_s < 10.0


class TestCircleContractFormula:
    """Verify quadratic formula directly."""

    def test_known_ttc(self):
        """Head-on: two circles, gap=10m, radius=2.236 each, closing=20m/s."""
        ego = _box(0, 0, 4, 2, 0, 10, 0)
        tgt = _box(20, 0, 4, 2, math.pi, -10, 0)
        r1 = ego.radius
        r2 = tgt.radius
        R = r1 + r2
        # gap = 20 - R, closing = 20
        expected_ttc = (20 - R) / 20.0
        result = compute_ttc_fast(ego, tgt)
        assert abs(result.derived_ttc_2d_s - expected_ttc) < 0.01


class TestDeprecatedAlias:
    """compute_ttc_obb should raise DeprecationWarning."""

    def test_alias_raises(self):
        ego = _box(0, 0, 4, 2, 0, 10, 0)
        tgt = _box(20, 0, 4, 2, 0, 0, 0)
        with pytest.raises(DeprecationWarning):
            compute_ttc_obb(ego, tgt)
