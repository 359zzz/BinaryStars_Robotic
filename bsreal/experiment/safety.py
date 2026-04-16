"""Safety mechanisms for real-robot experiments."""

from __future__ import annotations

import math
import time
import logging

logger = logging.getLogger(__name__)

# OpenArm joint limits (degrees) — from Evo-RL config, right arm
OPENARM_JOINT_LIMITS_DEG = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-9.0, 90.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
}

# Piper joint limits (degrees) — conservative defaults
PIPER_JOINT_LIMITS_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-90.0, 90.0),
    "joint_3": (-140.0, 10.0),
    "joint_4": (-170.0, 170.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-170.0, 170.0),
}

# Safety thresholds
MAX_POSITION_ERROR_DEG = 10.0   # emergency stop if exceeded
MAX_VELOCITY_DEG_S = 60.0       # emergency stop if exceeded
MAX_PERTURBATION_DEG = 5.0      # upper bound on amplitude


class SafetyError(Exception):
    """Raised when a safety limit is violated."""
    pass


def check_within_limits(positions_deg: dict[str, float]) -> None:
    """Verify all joint positions are within limits."""
    for name, pos in positions_deg.items():
        if name in OPENARM_JOINT_LIMITS_DEG:
            lo, hi = OPENARM_JOINT_LIMITS_DEG[name]
            if not (lo - 5.0 <= pos <= hi + 5.0):
                raise SafetyError(
                    f"{name} at {pos:.1f} deg outside limits [{lo}, {hi}]"
                )


def check_position_error(
    current_deg: dict[str, float],
    target_deg: dict[str, float],
    max_error: float = MAX_POSITION_ERROR_DEG,
) -> None:
    """Verify no joint has excessive position error."""
    for name in current_deg:
        if name in target_deg:
            err = abs(current_deg[name] - target_deg[name])
            if err > max_error:
                raise SafetyError(
                    f"{name} position error {err:.1f} deg > {max_error} deg"
                )


def _minimum_jerk(t: float) -> float:
    """Minimum-jerk profile: smooth acceleration and deceleration."""
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def slow_move(
    robot,
    target_deg: dict[str, float],
    duration_s: float = 3.0,
    dt: float = 0.02,
    custom_kp: dict[str, float] | None = None,
    custom_kd: dict[str, float] | None = None,
) -> None:
    """Smoothly move from current position to target (minimum-jerk profile)."""
    obs = robot.get_observation()
    start = {k: obs.get(k, 0.0) for k in target_deg}
    n_steps = max(int(duration_s / dt), 1)

    for step in range(n_steps + 1):
        alpha = _minimum_jerk(step / n_steps)
        cmd = {}
        for k in target_deg:
            cmd[k] = start[k] + alpha * (target_deg[k] - start[k])
        if custom_kp is not None or custom_kd is not None:
            robot.send_action(cmd, custom_kp=custom_kp, custom_kd=custom_kd)
        else:
            robot.send_action(cmd)
        time.sleep(dt)

    if custom_kp is not None or custom_kd is not None:
        robot.send_action(target_deg, custom_kp=custom_kp, custom_kd=custom_kd)
    else:
        robot.send_action(target_deg)
    time.sleep(0.3)


def emergency_freeze(robot) -> None:
    """Freeze robot at current position (send current pos as target)."""
    try:
        obs = robot.get_observation()
        freeze_cmd = {
            k: obs[k] for k in obs if k.endswith(".pos")
        }
        robot.send_action(freeze_cmd)
        logger.warning("EMERGENCY FREEZE: holding current position")
    except Exception as e:
        logger.error(f"Failed to freeze: {e}")
