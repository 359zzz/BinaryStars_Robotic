"""Sinusoidal perturbation experiment core.

Applies a sinusoidal position command to one joint while holding all others,
and records the torque/position/velocity response at all joints.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field

import numpy as np

from bsreal.experiment.safety import (
    SafetyError,
    check_position_error,
    emergency_freeze,
    MAX_POSITION_ERROR_DEG,
)

logger = logging.getLogger(__name__)


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

    logger.info(
        f"Perturbation: {perturb_joint}, A={config.amplitude_deg} deg, "
        f"f={config.frequency_hz} Hz, T={config.duration_s} s"
    )

    t0 = time.monotonic()
    n_errors = 0

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

            robot.send_action(cmd)

            # Read observation
            obs = robot.get_observation()

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
        robot.send_action(cmd)

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
