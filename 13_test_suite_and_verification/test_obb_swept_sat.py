#!/usr/bin/env python3
"""
Test Swept-OBB SAT — 10 deterministic scenarios + dense comparison.
"""
import math
import pytest
import numpy as np
from phase2_womd.obb_ttc_swept import (
    OBBAgent, compute_ttc_obb_swept, compute_ttc_obb_dense_sat, T_MAX_S,
)

TOL = 0.05  # 50ms tolerance for analytical vs dense comparison


def _agent(cx, cy, l, w, h, vx, vy):
    return OBBAgent(cx=cx, cy=cy, length=l, width=w, heading=h, vx=vx, vy=vy)


# ---------- 10 deterministic scenarios ----------

class TestSweptOBBDeterministic:

    def test_01_current_overlap(self):
        """Two overlapping boxes → TTC=0, current_geometry_overlap."""
        ego = _agent(0, 0, 4, 2, 0, 0, 0)
        tgt = _agent(1, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.pair_valid
        assert r.overlap_now
        assert r.status == "current_geometry_overlap"
        assert r.ttc_s == 0.0

    def test_02_static_separated(self):
        """Two static boxes far apart → right_censored."""
        ego = _agent(0, 0, 4, 2, 0, 0, 0)
        tgt = _agent(100, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.status == "right_censored"
        assert r.ttc_s == T_MAX_S

    def test_03_rear_end_closing(self):
        """Ego closing on stationary target ahead."""
        ego = _agent(0, 0, 4, 2, 0, 10, 0)
        tgt = _agent(20, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.hit_future
        assert r.status == "future_contact_event"
        # Gap = 20 - 2 - 2 = 16m at 10m/s → ~1.6s
        assert 1.0 < r.ttc_s < 2.0

    def test_04_head_on(self):
        """Head-on collision."""
        ego = _agent(0, 0, 4, 2, 0, 10, 0)
        tgt = _agent(30, 0, 4, 2, math.pi, -10, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.hit_future
        assert r.status == "future_contact_event"
        # Gap = 30 - 4 = 26m at 20 m/s relative → ~1.3s
        assert 0.5 < r.ttc_s < 2.0

    def test_05_diverging(self):
        """Two boxes moving apart → right_censored."""
        ego = _agent(0, 0, 4, 2, 0, -10, 0)
        tgt = _agent(20, 0, 4, 2, 0, 10, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.status == "right_censored"

    def test_06_adjacent_parallel(self):
        """Side-by-side, same speed, no lateral closure → right_censored."""
        ego = _agent(0, 0, 4, 2, 0, 10, 0)
        tgt = _agent(0, 5, 4, 2, 0, 10, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.status == "right_censored"

    def test_07_perpendicular_crossing(self):
        """90-degree intersection collision."""
        ego = _agent(0, -30, 4, 2, math.pi / 2, 0, 10)
        tgt = _agent(-30, 0, 4, 2, 0, 10, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.hit_future
        assert r.status == "future_contact_event"
        # Both arrive at origin at ~3s
        assert 2.0 < r.ttc_s < 4.0

    def test_08_glancing_contact(self):
        """Near-miss: offset enough to just barely touch."""
        ego = _agent(0, 0, 4, 2, 0, 10, 0)
        tgt = _agent(20, 1.9, 4, 2, 0, 0, 0)  # lateral offset just inside
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.hit_future
        assert r.status == "future_contact_event"

    def test_09_vehicle_pedestrian(self):
        """Vehicle approaching small pedestrian."""
        ego = _agent(0, 0, 4.5, 2.0, 0, 8, 0)
        tgt = _agent(15, 0.5, 0.5, 0.5, 0, 0, 1)  # ped crossing
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.pair_valid
        # Should either hit or miss depending on timing
        assert r.status in ("future_contact_event", "right_censored")

    def test_10_vehicle_cyclist(self):
        """Vehicle overtaking cyclist."""
        ego = _agent(0, 0, 4.5, 2.0, 0, 15, 0)
        tgt = _agent(30, 0.3, 1.8, 0.6, 0, 5, 0)  # cyclist ahead
        r = compute_ttc_obb_swept(ego, tgt)
        assert r.pair_valid
        if r.hit_future:
            assert r.status == "future_contact_event"
            # Gap = 30 - 2.25 - 0.9 ≈ 26.85m at 10m/s → ~2.7s
            assert 1.5 < r.ttc_s < 4.0


# ---------- Dense SAT comparison ----------

class TestSweptVsDenseComparison:

    @pytest.mark.parametrize("seed", range(50))
    def test_random_pair_swept_vs_dense(self, seed):
        """Random valid pair: swept-SAT matches dense-SAT within tolerance."""
        rng = np.random.RandomState(seed)

        # Random scenario
        ego = _agent(
            cx=0, cy=0,
            l=rng.uniform(3, 6), w=rng.uniform(1.5, 2.5),
            h=rng.uniform(-math.pi, math.pi),
            vx=rng.uniform(-15, 15), vy=rng.uniform(-15, 15)
        )
        tgt = _agent(
            cx=rng.uniform(-50, 50), cy=rng.uniform(-50, 50),
            l=rng.uniform(0.5, 6), w=rng.uniform(0.3, 2.5),
            h=rng.uniform(-math.pi, math.pi),
            vx=rng.uniform(-15, 15), vy=rng.uniform(-15, 15)
        )

        swept = compute_ttc_obb_swept(ego, tgt)
        dense = compute_ttc_obb_dense_sat(ego, tgt, dt=0.001)

        # Both should agree on hit/no-hit
        if dense.hit_future:
            assert swept.hit_future, (
                f"Seed {seed}: dense found contact at {dense.ttc_s:.4f}s "
                f"but swept says right_censored"
            )
            # Swept should be <= dense + tolerance (analytical is exact or earlier)
            assert swept.ttc_s <= dense.ttc_s + TOL, (
                f"Seed {seed}: swept={swept.ttc_s:.4f} > dense={dense.ttc_s:.4f}+tol"
            )
        else:
            # If dense says no hit, swept can still say no hit
            # (dense might miss due to step size, swept is exact)
            pass

        # If swept says contact, dense should agree (within step resolution)
        if swept.hit_future and not dense.hit_future:
            # swept found contact that dense missed — check if near horizon
            # This can happen if contact is very brief (tangent)
            assert swept.ttc_s > T_MAX_S - 0.1 or (
                swept.contact_exit - swept.contact_entry < 0.002
            ), f"Seed {seed}: swept found contact at {swept.ttc_s:.4f}s but dense missed"


class TestInvalidInputs:

    def test_invalid_ego(self):
        ego = _agent(0, 0, 4, 2, 0, 10, 0)
        ego.valid = False
        tgt = _agent(10, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert not r.pair_valid
        assert r.status == "invalid"

    def test_zero_dimensions(self):
        ego = _agent(0, 0, 0, 2, 0, 10, 0)
        tgt = _agent(10, 0, 4, 2, 0, 0, 0)
        r = compute_ttc_obb_swept(ego, tgt)
        assert not r.pair_valid
