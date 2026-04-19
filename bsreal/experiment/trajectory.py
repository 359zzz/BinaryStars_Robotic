"""Bimanual trajectory definitions for coordination experiments.

Supports both OpenArm (7-DOF) and Piper (6-DOF).
Config names starting with "piper_" use 6-DOF settings; others use 7-DOF.

Bar-holding protocol: both arms extend forward, grippers hold a rigid bar.
Trajectory only varies "safe" axes (won't drop bar).

OpenArm safe axes: j1 (shoulder yaw), j4 (elbow), j6 (wrist rot)
Piper safe axes:   j2, j3, j5
"""

from __future__ import annotations

import numpy as np

# ── Task object definitions ──────────────────────────────────────────────────

TASK_OBJECTS = {
    "independent":  {"mass": 0.0,  "geometry": "none",     "dims": ()},
    "bar_only":     {"mass": 0.3,  "geometry": "cylinder",  "dims": (0.02, 0.25)},
    "bar_loaded":   {"mass": 1.5,  "geometry": "cylinder",  "dims": (0.02, 0.25)},
}

# ── OpenArm configs (7-DOF) ─────────────────────────────────────────────────

_OPENARM_CONFIGS = {
    "bar_mid": {
        # Legacy alias kept for compatibility. This now matches the
        # validated front-lift bar grasp pose used in successful real runs.
        "joint_1": 30.0,
        "joint_2": 0.0,
        "joint_3": 0.0,
        "joint_4": 40.0,
        "joint_5": 75.0,
        "joint_6": 25.0,
        "joint_7": 0.0,
    },
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

_OPENARM_MIRROR = {"joint_1", "joint_2", "joint_3", "joint_5", "joint_6", "joint_7"}

_OPENARM_WAYPOINTS = [
    (0.00, {}),
    (0.25, {"joint_1": -8.0, "joint_4": 15.0, "joint_6": -8.0}),
    (0.50, {}),
    (0.75, {"joint_1": 8.0, "joint_4": -10.0, "joint_6": 8.0}),
    (1.00, {}),
]

# ── Piper configs (6-DOF) ───────────────────────────────────────────────────

_PIPER_CONFIGS = {
    "piper_a": {
        "joint_1": -25.0, "joint_2": 20.0, "joint_3": -25.0,
        "joint_4": -25.0, "joint_5": 0.0, "joint_6": -65.0,
    },
    "piper_b": {
        "joint_1": -25.0, "joint_2": 35.0, "joint_3": -25.0,
        "joint_4": -25.0, "joint_5": 0.0, "joint_6": -65.0,
    },
    "piper_c": {
        "joint_1": -25.0, "joint_2": 20.0, "joint_3": -40.0,
        "joint_4": -25.0, "joint_5": 0.0, "joint_6": -65.0,
    },
}

_PIPER_MIRROR = {"joint_1", "joint_4", "joint_6"}

_PIPER_WAYPOINTS = [
    (0.00, {}),
    (0.25, {"joint_2": 10.0, "joint_3": 8.0, "joint_5": -10.0}),
    (0.50, {}),
    (0.75, {"joint_2": -8.0, "joint_3": -8.0, "joint_5": 10.0}),
    (1.00, {}),
]

# ── Combined config dict (used by CLI) ──────────────────────────────────────

COORDINATION_CONFIGS = {**_OPENARM_CONFIGS, **_PIPER_CONFIGS}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_piper(config_name: str) -> bool:
    return config_name.startswith("piper_")


def _get_settings(config_name: str):
    """Return (n_joints, mirror_set, waypoints) for the given config."""
    if _is_piper(config_name):
        return 6, _PIPER_MIRROR, _PIPER_WAYPOINTS
    return 7, _OPENARM_MIRROR, _OPENARM_WAYPOINTS


def _minimum_jerk(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _interpolate_waypoints(
    waypoints: list[tuple[float, dict[str, float]]],
    t_frac: float,
) -> dict[str, float]:
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


# ── Public API ───────────────────────────────────────────────────────────────

def generate_bimanual_trajectory(
    config_name: str,
    duration_s: float = 10.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate bimanual trajectory.

    Returns
    -------
    timestamps : (T,) array
    q_right : (T, n_joints) degrees
    q_left  : (T, n_joints) degrees
    """
    n_joints, mirror, waypoints = _get_settings(config_name)
    base = COORDINATION_CONFIGS[config_name]
    joint_names = [f"joint_{i}" for i in range(1, n_joints + 1)]
    n_steps = int(duration_s / dt) + 1
    timestamps = np.linspace(0.0, duration_s, n_steps)

    q_right = np.zeros((n_steps, n_joints))
    q_left = np.zeros((n_steps, n_joints))

    for step, t in enumerate(timestamps):
        t_frac = t / duration_s
        delta = _interpolate_waypoints(waypoints, t_frac)

        for j, jn in enumerate(joint_names):
            base_val = base[jn]
            d = delta.get(jn, 0.0)
            q_right[step, j] = base_val + d
            if jn in mirror:
                q_left[step, j] = -(base_val + d)
            else:
                q_left[step, j] = base_val + d

    return timestamps, q_right, q_left


def generate_bimanual_trajectory_rad(
    config_name: str,
    duration_s: float = 10.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Same but returns (T, 2*n_joints) in radians."""
    timestamps, q_right_deg, q_left_deg = generate_bimanual_trajectory(
        config_name, duration_s, dt,
    )
    q_full_deg = np.hstack([q_right_deg, q_left_deg])
    q_full_rad = np.deg2rad(q_full_deg)
    return timestamps, q_full_rad


def get_start_positions_deg(config_name: str) -> dict[str, float]:
    """Start position dict for all joints (LeRobot format)."""
    n_joints, mirror, _ = _get_settings(config_name)
    base = COORDINATION_CONFIGS[config_name]
    positions = {}
    for i in range(1, n_joints + 1):
        jn = f"joint_{i}"
        val = base[jn]
        positions[f"right_joint_{i}"] = val
        if jn in mirror:
            positions[f"left_joint_{i}"] = -val
        else:
            positions[f"left_joint_{i}"] = val
    return positions
