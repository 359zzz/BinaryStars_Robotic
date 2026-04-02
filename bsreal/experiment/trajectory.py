"""Bimanual trajectory definitions for coordination experiments.

Bar-holding protocol: both arms extend forward with wrists up, grippers
hold a rigid aluminum bar.  Trajectory varies j1/j4/j6 (safe axes for
bar holding).  j2/j3/j5/j7 stay fixed to avoid dropping the bar.

Joint roles (OpenArm v10):
  joint_1: shoulder yaw (rotate arm inward/outward)
  joint_2: shoulder abduction (sideways lift) — keep near 0 for bar!
  joint_3: shoulder roll
  joint_4: elbow bend
  joint_5: wrist pitch (75 deg = wrist standing up)
  joint_6: wrist rotation
  joint_7: wrist roll
"""

from __future__ import annotations

import numpy as np

# ── Task object definitions ──────────────────────────────────────────────────

TASK_OBJECTS = {
    "independent":  {"mass": 0.0,  "geometry": "none",     "dims": ()},
    "bar_only":     {"mass": 0.3,  "geometry": "cylinder",  "dims": (0.02, 0.25)},
    "bar_loaded":   {"mass": 1.5,  "geometry": "cylinder",  "dims": (0.02, 0.25)},
}

# ── Arm configurations (degrees) ────────────────────────────────────────────
# Single arm (right) angles.  Left arm mirrors joints 1,2,3,5,6,7.
# Base pose from hardware testing: arms forward, wrists up, grippers inward.

COORDINATION_CONFIGS = {
    "bar_a": {
        "joint_1": 30.0, "joint_2": 0.0, "joint_3": 0.0, "joint_4": 20.0,
        "joint_5": 75.0, "joint_6": 25.0, "joint_7": 0.0,
    },
    "bar_b": {
        "joint_1": 30.0, "joint_2": 0.0, "joint_3": 0.0, "joint_4": 40.0,
        "joint_5": 75.0, "joint_6": 25.0, "joint_7": 0.0,
    },
    "bar_c": {
        "joint_1": 20.0, "joint_2": 0.0, "joint_3": 0.0, "joint_4": 30.0,
        "joint_5": 75.0, "joint_6": 25.0, "joint_7": 0.0,
    },
}

# ── Trajectory waypoints ────────────────────────────────────────────────────
# Only vary safe axes: joint_1 (shoulder yaw), joint_4 (elbow), joint_6 (wrist rot).
# These can change without dropping the bar (confirmed by user testing).

_BAR_WAYPOINTS = [
    (0.00, {}),                                                        # start
    (0.25, {"joint_1": -8.0, "joint_4": 15.0, "joint_6": -8.0}),      # inward + bend
    (0.50, {}),                                                        # center
    (0.75, {"joint_1": 8.0, "joint_4": -10.0, "joint_6": 8.0}),       # outward + extend
    (1.00, {}),                                                        # return
]

# Joints whose sign flips for left arm
_MIRROR_JOINTS = {"joint_1", "joint_2", "joint_3", "joint_5", "joint_6", "joint_7"}


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

    for step, t in enumerate(timestamps):
        t_frac = t / duration_s
        delta = _interpolate_waypoints(_BAR_WAYPOINTS, t_frac)

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
