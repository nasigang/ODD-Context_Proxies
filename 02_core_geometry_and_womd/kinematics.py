#!/usr/bin/env python3
"""
Derived Kinematics from Agent State
=====================================
Finite-difference computations over consecutive valid states ONLY.

Rules enforced:
  1. Invalid states are NEVER filled with 0 — they stay NaN.
  2. Finite differences require consecutive valid states.
  3. Gap / boundary frames → derived_kinematic_valid = False.
  4. Heading wrapping to [-π, π) for yaw rate.

Columns produced:
  derived_speed_mps       — √(vx² + vy²), NaN when valid=False
  derived_accel_mps2      — Δspeed / Δt, needs 2 consecutive valid
  derived_yaw_rate_rps    — Δheading / Δt (wrapped), needs 2 consecutive valid
  derived_jerk_mps3       — Δaccel / Δt, needs 3 consecutive valid
  derived_kinematic_valid — True only when ALL derived values are finite
"""

import math

import numpy as np
import pandas as pd


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived kinematic columns to an agent_state DataFrame.

    The DataFrame must contain:
      scenario_id, track_id, time_index, valid,
      velocity_x, velocity_y, heading, timestamp_seconds

    Operates per (scenario_id, track_id) group, sorted by time_index.
    Returns the same DataFrame with 5 new derived_* columns.
    """
    derived_cols = [
        "derived_speed_mps",
        "derived_accel_mps2",
        "derived_yaw_rate_rps",
        "derived_jerk_mps3",
        "derived_kinematic_valid",
    ]
    # Pre-allocate with NaN / False
    for col in derived_cols[:-1]:
        df[col] = np.nan
    df["derived_kinematic_valid"] = False

    groups = df.groupby(["scenario_id", "track_id"], sort=False)
    for _key, idx in groups.groups.items():
        sub = df.loc[idx].sort_values("time_index")
        result = _compute_group(sub)
        for col in derived_cols:
            df.loc[sub.index, col] = result[col].values

    return df


def _compute_group(g: pd.DataFrame) -> pd.DataFrame:
    """Compute derived kinematics for a single (scenario_id, track_id) group."""
    n = len(g)
    speed = np.full(n, np.nan, dtype=np.float64)
    accel = np.full(n, np.nan, dtype=np.float64)
    yaw_rate = np.full(n, np.nan, dtype=np.float64)
    jerk = np.full(n, np.nan, dtype=np.float64)
    kin_valid = np.zeros(n, dtype=bool)

    valid = g["valid"].values.astype(bool)
    vx = g["velocity_x"].values.astype(np.float64)
    vy = g["velocity_y"].values.astype(np.float64)
    heading = g["heading"].values.astype(np.float64)
    ts = g["timestamp_seconds"].values.astype(np.float64)

    # --- Step 1: speed (requires valid only) ---
    valid_mask = valid & np.isfinite(vx) & np.isfinite(vy)
    speed[valid_mask] = np.sqrt(vx[valid_mask] ** 2 + vy[valid_mask] ** 2)

    # --- Step 2: accel & yaw_rate (requires 2 consecutive valid) ---
    for i in range(1, n):
        if valid_mask[i] and valid_mask[i - 1]:
            dt = ts[i] - ts[i - 1]
            if dt > 0:
                accel[i] = (speed[i] - speed[i - 1]) / dt
                dh = _wrap_angle(heading[i] - heading[i - 1])
                yaw_rate[i] = dh / dt

    # --- Step 3: jerk (requires 3 consecutive valid → 2 consecutive accel) ---
    for i in range(2, n):
        if (valid_mask[i] and valid_mask[i - 1] and valid_mask[i - 2]
                and np.isfinite(accel[i]) and np.isfinite(accel[i - 1])):
            dt = ts[i] - ts[i - 1]
            if dt > 0:
                jerk[i] = (accel[i] - accel[i - 1]) / dt

    # --- Step 4: kinematic valid mask ---
    kin_valid = (
        np.isfinite(speed)
        & np.isfinite(accel)
        & np.isfinite(yaw_rate)
        & np.isfinite(jerk)
    )

    g = g.copy()
    g["derived_speed_mps"] = speed
    g["derived_accel_mps2"] = accel
    g["derived_yaw_rate_rps"] = yaw_rate
    g["derived_jerk_mps3"] = jerk
    g["derived_kinematic_valid"] = kin_valid
    return g


def _wrap_angle(angle: float) -> float:
    """Wrap angle difference to [-π, π)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
