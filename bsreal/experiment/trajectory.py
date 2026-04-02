"""Bimanual trajectory definitions for coordination experiments.

4 tasks share the SAME trajectory (paper: fixed trajectory, vary object only).
5 arm configurations define different M(q) coupling landscapes.
"""

from __future__ import annotations

import math
import numpy as np

# ── Task object definitions ──────────────────────────────────────────────────

TASK_OBJECTS = {
    "independent":  {"mass": 0.0,  "geometry": "none",     "dims": ()},
    "box_lift":     {"mass": 0.2,  "geometry": "box",      "dims": (0.15, 0.10, 0.10)},
    "barbell_lift": {"mass": 2.0,  "geometry": "cylinder",  "dims": (0.02, 0.40)},
    "rod_rotation": {"mass": 1.0,  "geometry": "cylinder",  "dims": (0.015, 0.30)},
}

# ── Arm configurations (degrees) ────────────────────────────────────────────
# Each config is {joint_name: angle_deg} for a SINGLE arm (7-DOF OpenArm).
# Dual-arm = right uses config as-is, left mirrors sign of joints 1,3,5,7.

COORDINATION_CONFIGS = {
    "home": {
        "joint_1": 0.0, "joint_2": 45.0, "joint_3": 0.0, "joint_4": 67.5,
        "joint_5": 0.0, "joint_6": 0.0, "joint_7": 0.0,
    },
    "elbow_up": {
        "joint_1": 0.0, "joint_2": 30.0, "joint_3": 0.0, "joint_4": 120.0,
        "joint_5": 0.0, "joint_6": 0.0, "joint_7": 0.0,
    },
    "shoulder_elbow": {
        "joint_1": 20.0, "joint_2": 60.0, "joint_3": -30.0, "joint_4": 90.0,
        "joint_5": 0.0, "joint_6": 0.0, "joint_7": 0.0,
    },
    "full_reach": {
        "joint_1": 0.0, "joint_2": 80.0, "joint_3": 0.0, "joint_4": 45.0,
        "joint_5": 0.0, "joint_6": 20.0, "joint_7": 0.0,
    },
    "wrist_engaged": {
        "joint_1": 0.0, "joint_2": 45.0, "joint_3": 0.0, "joint_4": 90.0,
        "joint_5": 30.0, "joint_6": -20.0, "joint_7": 40.0,
    },
}

# ── Trajectory waypoints (degrees, relative to start config) ────────────────
# Each trajectory is a list of (time_fraction, delta_q_deg) pairs.
# delta_q_deg is a dict of {joint_name: delta_angle} for the RIGHT arm.
# Left arm mirrors: joints 1,3,5,7 negate delta.

_TRAJECTORY_WAYPOINTS = [
    (0.0,  {}),  # start
    (0.25, {"joint_2": 10.0, "joint_4": -15.0}),
    (0.50, {"joint_2": 15.0, "joint_4": -25.0, "joint_5": 10.0}),
    (0.75, {"joint_2": 10.0, "joint_4": -15.0}),
    (1.0,  {}),  # return to start
]

# Joints whose sign flips for left arm (odd-indexed in kinematic chain)
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

    Parameters
    ----------
    config_name : key into COORDINATION_CONFIGS
    duration_s : total duration in seconds
    dt : timestep

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

    for step, t in enumerate(timestamps):
        t_frac = t / duration_s
        delta = _interpolate_waypoints(_TRAJECTORY_WAYPOINTS, t_frac)

        for j, jn in enumerate(joint_names):
            base_val = base[jn]
            d = delta.get(jn, 0.0)
            q_right[step, j] = base_val + d
            # Left arm: mirror odd joints
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
