#!/usr/bin/env python3
"""
Scenario Diagnostics Tests
==============================
Tests for R_t/S_t computation, hysteresis event detection, recovery,
and D_s^risk diagnostic vector.

Required toy tests:
  1. Constant-zero R_t → peak=0, event_count=0
  2. Single spike → peak correct, event_count=1
  3. Two events with gap < merge_gap → merged to 1
  4. Two events with gap > merge_gap → count=2
  5. Recovery before scenario end → recovery_time > 0
  6. No recovery → recovery_censored=True
  7. S_t only on event frames
  8. Timestamp-based duration (not frame-count)
"""

import math

import numpy as np
import pandas as pd
import pytest

from phase2_womd.diag_config import DiagConfig
from phase2_womd.scenario_diagnostics import (
    compute_Rt,
    compute_St,
    compute_diagnostic_vector,
    compute_recovery,
    detect_exceedance_events,
    compute_all_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, dt: float = 0.1) -> np.ndarray:
    """Create timestamps starting at 0 with given spacing."""
    return np.arange(n) * dt


def _default_cfg(**overrides) -> DiagConfig:
    return DiagConfig(**overrides)


# ---------------------------------------------------------------------------
# Test 1: Constant-zero R_t
# ---------------------------------------------------------------------------
class TestConstantZero:
    def test_zero_Rt_peak_zero(self):
        n = 50
        Rt = np.zeros(n)
        St = np.full(n, np.nan)
        ts = _make_timestamps(n)
        mu = np.full(n, 1.0)
        sigma = np.full(n, 0.5)

        dvec = compute_diagnostic_vector(Rt, St, ts, mu, sigma, _default_cfg())
        assert dvec["peak"] == 0.0
        assert dvec["event_count"] == 0
        assert dvec["exceedance_duration"] == 0.0
        assert dvec["max_event_duration"] == 0.0
        assert dvec["recovery_censored"] is False


# ---------------------------------------------------------------------------
# Test 2: Single spike
# ---------------------------------------------------------------------------
class TestSingleSpike:
    def test_single_spike_detected(self):
        """R_t = 0 except frames 20–30 where R_t = 0.9."""
        n = 50
        Rt = np.zeros(n)
        Rt[20:31] = 0.9
        St = np.full(n, np.nan)
        ts = _make_timestamps(n, dt=0.1)
        mu = np.full(n, 1.0)
        sigma = np.full(n, 0.5)

        cfg = _default_cfg(theta_on=0.7, theta_off=0.3,
                           minimum_event_duration_s=0.3)
        dvec = compute_diagnostic_vector(Rt, St, ts, mu, sigma, cfg)

        assert dvec["peak"] == pytest.approx(0.9, abs=1e-6)
        assert dvec["event_count"] == 1
        assert dvec["max_event_duration"] > 0


# ---------------------------------------------------------------------------
# Test 3: Two events, gap < merge_gap → merged
# ---------------------------------------------------------------------------
class TestMergeEvents:
    def test_merge_close_events(self):
        """Two exceedances separated by 0.3s gap (< merge_gap=0.5)."""
        n = 100
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        # Event 1: frames 10-20 (1.0s-2.0s)
        Rt[10:21] = 0.8
        # Gap: frames 21-23 (0.3s gap)
        # Event 2: frames 24-34 (2.4s-3.4s)
        Rt[24:35] = 0.8

        cfg = _default_cfg(merge_gap_s=0.5, minimum_event_duration_s=0.3)
        events = detect_exceedance_events(Rt, ts, cfg)

        # Should merge to 1 event
        assert len(events) == 1

    def test_no_merge_distant_events(self):
        """Two exceedances separated by 2.0s gap (> merge_gap=0.5)."""
        n = 100
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        # Event 1: frames 10-15
        Rt[10:16] = 0.8
        # Gap: frames 16-35 (2.0s gap)
        # Event 2: frames 36-45
        Rt[36:46] = 0.8

        cfg = _default_cfg(merge_gap_s=0.5, minimum_event_duration_s=0.3)
        events = detect_exceedance_events(Rt, ts, cfg)

        # Should be 2 separate events
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Test 4: Two events, gap > merge_gap → count=2
# ---------------------------------------------------------------------------
class TestTwoDistinctEvents:
    def test_distinct_events_counted(self):
        n = 100
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        Rt[5:15] = 0.9    # event 1
        Rt[50:65] = 0.85   # event 2

        cfg = _default_cfg(merge_gap_s=0.5, minimum_event_duration_s=0.3)
        St = np.full(n, np.nan)
        mu = np.full(n, 1.0)
        sigma = np.full(n, 0.5)

        dvec = compute_diagnostic_vector(Rt, St, ts, mu, sigma, cfg)
        assert dvec["event_count"] == 2


# ---------------------------------------------------------------------------
# Test 5: Recovery before scenario end
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_recovery_detected(self):
        """Event ends, then R_t stays below theta_off for recovery_hold_time."""
        n = 100
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        Rt[10:20] = 0.9  # event at 1.0-2.0s

        cfg = _default_cfg(recovery_hold_time_s=0.5)
        events = detect_exceedance_events(Rt, ts, cfg)

        recovery_time, censored = compute_recovery(Rt, ts, events, cfg)
        assert recovery_time > 0
        assert censored is False


# ---------------------------------------------------------------------------
# Test 6: No recovery → censored
# ---------------------------------------------------------------------------
class TestRecoveryCensored:
    def test_no_recovery_censored(self):
        """Event runs until end of scenario."""
        n = 30
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        Rt[20:] = 0.9  # event starts at 2.0s, runs to end at 2.9s

        cfg = _default_cfg(
            recovery_hold_time_s=0.5,
            minimum_event_duration_s=0.3,
        )
        events = detect_exceedance_events(Rt, ts, cfg)

        recovery_time, censored = compute_recovery(Rt, ts, events, cfg)
        assert censored is True


# ---------------------------------------------------------------------------
# Test 7: S_t only on event frames
# ---------------------------------------------------------------------------
class TestSurprise:
    def test_St_only_event_frames(self):
        """S_t should be NaN for non-event frames."""
        n = 10
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        mu = np.full(n, 1.0)
        sigma = np.full(n, 0.5)
        exp_flag = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        status = np.array([
            "event", "right_censored", "event", "no_exposure",
            "no_exposure", "event", "right_censored",
            "invalid_ego_state", "invalid_frame", "event",
        ])

        St = compute_St(y, mu, sigma, exp_flag, status)

        # Only frames 0, 2, 5, 9 should have S_t (exposure=1 AND event)
        assert np.isfinite(St[0])
        assert np.isnan(St[1])   # right_censored
        assert np.isfinite(St[2])
        assert np.isnan(St[3])   # no_exposure
        assert np.isnan(St[4])   # no_exposure
        assert np.isfinite(St[5])
        assert np.isnan(St[6])   # right_censored
        assert np.isnan(St[7])   # invalid
        assert np.isnan(St[8])   # invalid
        assert np.isfinite(St[9])

    def test_St_positive(self):
        """S_t = -log(F) should be ≥ 0 when F ≤ 1."""
        y = np.array([0.5])
        mu = np.array([1.0])
        sigma = np.array([0.5])
        exp_flag = np.array([1])
        status = np.array(["event"])

        St = compute_St(y, mu, sigma, exp_flag, status)
        assert St[0] >= 0


# ---------------------------------------------------------------------------
# Test 8: Timestamp-based duration
# ---------------------------------------------------------------------------
class TestTimestampDuration:
    def test_variable_dt(self):
        """Exceedance duration uses actual timestamps, not frame count."""
        # Non-uniform timestamps: first 10 at 0.1s, then 10 at 0.5s
        ts = np.concatenate([
            np.arange(10) * 0.1,      # 0.0 to 0.9s
            0.9 + np.arange(1, 11) * 0.5,  # 1.4 to 5.9s
        ])
        n = len(ts)
        Rt = np.zeros(n)
        # Exceedance in later (longer-dt) frames
        Rt[12:16] = 0.9  # 4 frames at 0.5s spacing = 2.0s

        St = np.full(n, np.nan)
        mu = np.full(n, 1.0)
        sigma = np.full(n, 0.5)

        cfg = _default_cfg(
            theta_on=0.7, theta_off=0.3,
            minimum_event_duration_s=0.3,
        )
        dvec = compute_diagnostic_vector(Rt, St, ts, mu, sigma, cfg)

        # Exceedance duration should be ~2.0s (4 frames × 0.5s), not 0.4s
        assert dvec["exceedance_duration"] > 1.0

    def test_uniform_dt_consistent(self):
        """With uniform 0.1s timestamps, 10 exceedance frames = 1.0s."""
        n = 50
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        Rt[20:31] = 0.9  # 11 frames → 10 intervals of 0.1s = 1.0s

        St = np.full(n, np.nan)
        mu = np.full(n, 1.0)
        sigma = np.full(n, 0.5)

        cfg = _default_cfg(theta_on=0.7, theta_off=0.3)
        dvec = compute_diagnostic_vector(Rt, St, ts, mu, sigma, cfg)

        assert abs(dvec["exceedance_duration"] - 1.0) < 0.2


# ---------------------------------------------------------------------------
# Test: Minimum event duration filter
# ---------------------------------------------------------------------------
class TestMinimumDuration:
    def test_short_event_discarded(self):
        """Event shorter than min_duration is not counted."""
        n = 50
        ts = _make_timestamps(n, dt=0.1)
        Rt = np.zeros(n)
        Rt[20:22] = 0.9  # 2 frames = 0.1s → below min_duration=0.5

        cfg = _default_cfg(minimum_event_duration_s=0.5)
        events = detect_exceedance_events(Rt, ts, cfg)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test: R_t computation
# ---------------------------------------------------------------------------
class TestRtComputation:
    def test_Rt_zero_when_no_exposure(self):
        p_e = np.array([0.0])
        mu = np.array([1.0])
        sigma = np.array([0.5])
        Rt = compute_Rt(p_e, mu, sigma, tau=3.0)
        assert Rt[0] == 0.0

    def test_Rt_bounded_0_1(self):
        p_e = np.array([0.8])
        mu = np.array([1.0])
        sigma = np.array([0.5])
        Rt = compute_Rt(p_e, mu, sigma, tau=3.0)
        assert 0 <= Rt[0] <= 1.0


# ---------------------------------------------------------------------------
# Integration test: compute_all_diagnostics
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_batch_computation(self):
        """Create minimal frame predictions and run full pipeline."""
        rows = []
        for t in range(20):
            rows.append({
                "scenario_id": "s1",
                "time_index": t,
                "timestamp_seconds": t * 0.1,
                "pred_p_exposure": 0.8,
                "pred_mu": 1.0,
                "pred_sigma": 0.5,
                "y_log_ttc": 0.5 if t == 5 else 2.3,
                "ttc_censored": False if t == 5 else True,
                "exposure_flag": 1,
                "target_status": "event" if t == 5 else "right_censored",
                "split": "train",
            })
        df = pd.DataFrame(rows)

        vecs, curves = compute_all_diagnostics(df)

        assert len(vecs) == 1
        assert len(curves) == 20
        assert "peak" in vecs.columns
        assert "exceedance_duration" in vecs.columns
        assert "Rt" in curves.columns
        assert "St" in curves.columns
