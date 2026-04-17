"""Sinusoidal perturbation experiment core.

Applies a sinusoidal position command to one joint while holding all others,
and records the torque/position/velocity response at all joints.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from bsreal.experiment.safety import (
    SafetyError,
    check_position_error,
    emergency_freeze,
    MAX_POSITION_ERROR_DEG,
)

logger = logging.getLogger(__name__)

GRIPPER_HOLD_KP = 8.0
GRIPPER_HOLD_KD = 0.2
CAN_BUFFER_ERRNO = 105
CAN_RETRY_DELAYS_S = (0.02, 0.05, 0.1, 0.2)


@dataclass
class PerturbationConfig:
    """Parameters for a single perturbation trial."""

    amplitude_deg: float = 3.0      # sinusoidal amplitude
    frequency_hz: float = 0.5       # perturbation frequency
    duration_s: float = 10.0        # total duration (5 full periods at 0.5 Hz)
    ramp_s: float = 2.0             # amplitude ramp-up time
    dt: float = 0.01                # target sample interval (100 Hz)
    max_pos_error_deg: float = MAX_POSITION_ERROR_DEG


@dataclass
class TrialData:
    """Recorded data from one perturbation trial."""

    timestamps_s: list[float] = field(default_factory=list)
    positions_deg: list[list[float]] = field(default_factory=list)
    velocities_deg_s: list[list[float]] = field(default_factory=list)
    torques_Nm: list[list[float]] = field(default_factory=list)
    commanded_deg: list[list[float]] = field(default_factory=list)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None
    return chain


def _is_transient_can_buffer_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        if isinstance(item, OSError) and getattr(item, "errno", None) == CAN_BUFFER_ERRNO:
            return True
        message = str(item)
        if "No buffer space available" in message:
            return True
        if "Error Code 105" in message:
            return True
    return False


def _call_with_can_retry(
    op: Callable[[], Any],
    *,
    label: str,
    max_attempts: int = 1 + len(CAN_RETRY_DELAYS_S),
) -> Any:
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return op()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_can_buffer_error(exc) or attempt >= max_attempts:
                raise
            delay_s = CAN_RETRY_DELAYS_S[min(attempt - 1, len(CAN_RETRY_DELAYS_S) - 1)]
            logger.warning(
                "%s hit transient CAN buffer saturation on attempt %d/%d; backing off %.3fs",
                label,
                attempt,
                max_attempts,
                delay_s,
            )
            time.sleep(delay_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed without raising an exception")


def _send_action_with_retry(
    robot,
    cmd: dict[str, float],
    *,
    custom_kp=None,
    custom_kd=None,
    label: str,
) -> None:
    def _send() -> None:
        robot.send_action(cmd, custom_kp=custom_kp, custom_kd=custom_kd)

    _call_with_can_retry(_send, label=label)


def _get_observation_with_retry(robot, *, label: str) -> dict[str, float]:
    def _read() -> dict[str, float]:
        return robot.get_observation()

    return _call_with_can_retry(_read, label=label)


def run_perturbation_trial(
    robot,
    joint_names: list[str],
    base_positions_deg: dict[str, float],
    perturb_joint: str,
    config: PerturbationConfig | None = None,
) -> TrialData:
    """Run one sinusoidal perturbation trial.

    Parameters
    ----------
    robot : lerobot Robot instance (connected, with get_observation/send_action)
    joint_names : ordered list of joint names, e.g. ["joint_1", ..., "joint_7"]
    base_positions_deg : target position for all joints (degrees)
    perturb_joint : which joint to perturb, e.g. "joint_1"
    config : perturbation parameters

    Returns
    -------
    TrialData with recorded time series
    """
    if config is None:
        config = PerturbationConfig()

    if perturb_joint not in base_positions_deg:
        raise ValueError(f"{perturb_joint} not in base_positions_deg")

    data = TrialData()
    omega = 2.0 * math.pi * config.frequency_hz
    base_val = base_positions_deg[perturb_joint]
    auxiliary_positions_deg = {
        name: value for name, value in base_positions_deg.items() if name not in joint_names
    }
    custom_kp = {"gripper": GRIPPER_HOLD_KP} if "gripper" in auxiliary_positions_deg else None
    custom_kd = {"gripper": GRIPPER_HOLD_KD} if "gripper" in auxiliary_positions_deg else None

    logger.info(
        f"Perturbation: {perturb_joint}, A={config.amplitude_deg} deg, "
        f"f={config.frequency_hz} Hz, T={config.duration_s} s"
    )

    t0 = time.monotonic()
    n_errors = 0
    t = 0.0

    try:
        while True:
            t = time.monotonic() - t0
            if t >= config.duration_s:
                break

            # Amplitude ramp: linearly increase over ramp_s
            if t < config.ramp_s:
                amp = config.amplitude_deg * (t / config.ramp_s)
            else:
                amp = config.amplitude_deg

            # Build command: all joints at base + perturb joint with sine
            cmd = {}
            for jn in joint_names:
                key = f"{jn}.pos"
                if jn == perturb_joint:
                    cmd[key] = base_val + amp * math.sin(omega * t)
                else:
                    cmd[key] = base_positions_deg[jn]
            for aux_name, aux_value in auxiliary_positions_deg.items():
                cmd[f"{aux_name}.pos"] = aux_value

            _send_action_with_retry(
                robot,
                cmd,
                custom_kp=custom_kp,
                custom_kd=custom_kd,
                label=f"{perturb_joint}:send_action",
            )

            # Read observation
            obs = _get_observation_with_retry(
                robot,
                label=f"{perturb_joint}:get_observation",
            )

            # Record data
            data.timestamps_s.append(t)
            data.positions_deg.append([obs.get(f"{jn}.pos", 0.0) for jn in joint_names])
            data.velocities_deg_s.append([obs.get(f"{jn}.vel", 0.0) for jn in joint_names])
            data.torques_Nm.append([obs.get(f"{jn}.torque", 0.0) for jn in joint_names])
            data.commanded_deg.append([cmd.get(f"{jn}.pos", 0.0) for jn in joint_names])

            # Safety check: position error
            current = {jn: obs.get(f"{jn}.pos", 0.0) for jn in joint_names}
            target = {jn: base_positions_deg[jn] for jn in joint_names}
            target[perturb_joint] = cmd[f"{perturb_joint}.pos"]
            try:
                check_position_error(
                    {jn: current[jn] for jn in joint_names},
                    {jn: target[jn] for jn in joint_names},
                    config.max_pos_error_deg,
                )
                n_errors = 0
            except SafetyError as e:
                n_errors += 1
                if n_errors >= 3:
                    logger.error(f"Safety triggered: {e}")
                    emergency_freeze(robot)
                    raise

            # Pace the loop
            elapsed = time.monotonic() - t0 - t
            sleep_time = config.dt - (time.monotonic() - t0 - t)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except SafetyError:
        raise
    except KeyboardInterrupt:
        logger.info("Trial interrupted by user")
    finally:
        # Return to base position
        cmd = {f"{jn}.pos": base_positions_deg[jn] for jn in joint_names}
        for aux_name, aux_value in auxiliary_positions_deg.items():
            cmd[f"{aux_name}.pos"] = aux_value
        try:
            _send_action_with_retry(
                robot,
                cmd,
                custom_kp=custom_kp,
                custom_kd=custom_kd,
                label=f"{perturb_joint}:return_to_base",
            )
        except Exception:
            logger.exception("Failed to return %s trial to base position cleanly", perturb_joint)

    logger.info(f"Trial complete: {len(data.timestamps_s)} samples in {t:.1f}s")
    return data


def compute_theoretical_coupling(
    ir, q_rad: np.ndarray
) -> dict:
    """Compute predicted coupling matrix at configuration q."""
    from bsreal.dynamics.mass_matrix import compute_mass_matrix
    from bsreal.dynamics.coupling import normalized_coupling_matrix, local_field_terms

    M = compute_mass_matrix(ir, q_rad)
    J = normalized_coupling_matrix(M)
    h = local_field_terms(M)
    return {
        "M_matrix": M.tolist(),
        "J_matrix": J.tolist(),
        "h_fields": h.tolist(),
    }
