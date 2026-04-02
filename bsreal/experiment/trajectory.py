"""Bimanual trajectory definitions for coordination experiments.

Experiment design: both arms hold a rigid aluminum bar between grippers.
Trajectory is sagittal-plane only (joint_2/joint_4) to maintain grasp.
3 arm configs create different coupling landscapes → different M(q) → different J_ij.

Joint limits (LeRobot OpenArm):
  Right: joint_2 (-9, 90), joint_4 (0, 135)
  Left:  joint_2 (-90, 9), joint_4 (0, 135)  [joint_2 is mirrored]
"""

from __future__ import annotations

import numpy as np

# ── Task object definitions ──────────────────────────────────────────────────
# "independent" = no bar.  Others = aluminum bar with different total mass.
# User should weigh actual bar+weight and adjust mass here if needed.

TASK_OBJECTS = {
    "independent":  {"mass": 0.0,  "geometry": "none",     "dims": ()},
    "bar_only":     {"mass": 0.3,  "geometry": "cylinder",  "dims": (0.02, 0.25)},
    "bar_loaded":   {"mass": 1.5,  "geometry": "cylinder",  "dims": (0.02, 0.25)},
}

# ── Arm configurations (degrees) ────────────────────────────────────────────
# Each config = single arm angles.  Dual-arm: right as-is, left mirrors.
# All configs: arms forward, suitable for holding horizontal bar.
# Wrist joints (5,6,7) default to 0 — adjust on-site if gripper orientation
# needs tuning for your bar setup.

COORDINATION_CONFIGS = {
    "bar_low": {
        "joint_1": 0.0, "joint_2": 50.0, "joint_3": 0.0, "joint_4": 100.0,
        "joint_5": 0.0, "joint_6": 0.0, "joint_7": 0.0,
    },
    "bar_mid": {
        "joint_1": 0.0, "joint_2": 70.0, "joint_3": 0.0, "joint_4": 80.0,
        "joint_5": 0.0, "joint_6": 0.0, "joint_7": 0.0,
    },
    "bar_high": {
        "joint_1": 0.0, "joint_2": 75.0, "joint_3": 0.0, "joint_4": 55.0,
        "joint_5": 0.0, "joint_6": 0.0, "joint_7": 0.0,
    },
}

# ── Trajectory waypoints ────────────────────────────────────────────────────
# Sagittal plane only: joint_2 (shoulder pitch) + joint_4 (elbow).
# One full forward-backward oscillation.  No lateral/wrist motion → bar safe.

_BAR_WAYPOINTS = [
    (0.00, {}),                                          # start
    (0.25, {"joint_2": 10.0, "joint_4": -8.0}),         # forward reach
    (0.50, {}),                                          # center
    (0.75, {"joint_2": -10.0, "joint_4": 8.0}),         # pull back
    (1.00, {}),                                          # return
]

# Joints whose sign flips for left arm (SDK mirrored limits)
_MIRROR_JOINTS = {"joint_1", "joint_2", "joint_3", "joint_5", "joint_7"}


def _minimum_jerk(t: float) -> float:
    """Minimum-jerk interpolation: 0->1 smooth."""
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _interpolate_waypoints(
    waypoints: list[tuple[float, dict[str, float]]],
    t_frac: float,
) -> dict[str, float]:
    """Interpolate between waypoint deltas at fractional time t_frac."""
    if t_frac <= waypoints[0][0]:
        return dict(waypoints[0][1])
    if t_frac >= waypoints[-1][0]:
        return dict(waypoints[-1][1])

    for k in range(len(waypoints) - 1):
        t0, d0 = waypoints[k]
        t1, d1 = waypoints[k + 1]
        if t0 <= t_frac <= t1:
            alpha = _minimum_jerk((t_frac - t0) / (t1 - t0))
            all_keys = set(d0.keys()) | set(d1.keys())
            result = {}
            for key in all_keys:
                v0 = d0.get(key, 0.0)
                v1 = d1.get(key, 0.0)
                result[key] = v0 + alpha * (v1 - v0)
            return result

    return {}


def generate_bimanual_trajectory(
    config_name: str,
    duration_s: float = 10.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate bimanual trajectory for OpenArm 14-DOF.

    Returns
    -------
    timestamps : (T,) array
    q_right : (T, 7) target joint angles in degrees for right arm
    q_left : (T, 7) target joint angles in degrees for left arm
    """
    base = COORDINATION_CONFIGS[config_name]
    joint_names = [f"joint_{i}" for i in range(1, 8)]
    n_steps = int(duration_s / dt) + 1
    timestamps = np.linspace(0.0, duration_s, n_steps)

    q_right = np.zeros((n_steps, 7))
    q_left = np.zeros((n_steps, 7))

    waypoints = _BAR_WAYPOINTS

    for step, t in enumerate(timestamps):
        t_frac = t / duration_s
        delta = _interpolate_waypoints(waypoints, t_frac)

        for j, jn in enumerate(joint_names):
            base_val = base[jn]
            d = delta.get(jn, 0.0)
            q_right[step, j] = base_val + d
            if jn in _MIRROR_JOINTS:
                q_left[step, j] = -(base_val + d)
            else:
                q_left[step, j] = base_val + d

    return timestamps, q_right, q_left


def generate_bimanual_trajectory_rad(
    config_name: str,
    duration_s: float = 10.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as generate_bimanual_trajectory but returns (T, 14) in radians."""
    timestamps, q_right_deg, q_left_deg = generate_bimanual_trajectory(
        config_name, duration_s, dt,
    )
    q_full_deg = np.hstack([q_right_deg, q_left_deg])
    q_full_rad = np.deg2rad(q_full_deg)
    return timestamps, q_full_rad


def get_start_positions_deg(config_name: str) -> dict[str, float]:
    """Get the start position dict for all 14 joints (LeRobot format)."""
    base = COORDINATION_CONFIGS[config_name]
    positions = {}
    for i in range(1, 8):
        jn = f"joint_{i}"
        val = base[jn]
        positions[f"right_joint_{i}"] = val
        if jn in _MIRROR_JOINTS:
            positions[f"left_joint_{i}"] = -val
        else:
            positions[f"left_joint_{i}"] = val
    return positions
