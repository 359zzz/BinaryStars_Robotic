#!/usr/bin/env python3
"""Experiment 4: Piper 6-DOF single-arm coupling verification.

Piper via lerobot only returns joint positions (no torque, no velocity).
We use POSITION COUPLING as proxy: perturb one joint with sinusoidal
commands, measure position oscillation at other joints.  Due to dynamic
coupling M_ij, the low-level controller cannot perfectly reject the
disturbance → the non-perturbed joints show measurable position deviation
at the perturbation frequency, proportional to coupling strength.
"""

import argparse
import json
import math
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.robot_data.piper_data import make_piper_single_arm_ir
from bsreal.experiment.perturbation import (
    PerturbationConfig,
    compute_theoretical_coupling,
)
from bsreal.experiment.safety import slow_move

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]

# Piper configurations (rad) — matching Paper I multi-config analysis
CONFIGS = {
    "home":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "elbow_down": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    "reach":      [0.0, 0.5, -0.8, 0.0, 0.5, 0.0],
    "wrist":      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


def run_position_coupling_trial(robot, joint_names, base_positions_deg,
                                 perturb_joint, config):
    """Perturbation trial recording positions (Piper has no torque)."""
    omega = 2.0 * math.pi * config.frequency_hz
    base_val = base_positions_deg[perturb_joint]

    timestamps = []
    positions = []
    commanded = []

    logger.info(
        f"Perturbation: {perturb_joint}, A={config.amplitude_deg} deg, "
        f"f={config.frequency_hz} Hz, T={config.duration_s} s"
    )

    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            if t >= config.duration_s:
                break

            amp = config.amplitude_deg * min(t / config.ramp_s, 1.0)

            # Build command — Piper requires ALL 6 joints
            cmd = {}
            for jn in joint_names:
                if jn == perturb_joint:
                    cmd[f"{jn}.pos"] = base_val + amp * math.sin(omega * t)
                else:
                    cmd[f"{jn}.pos"] = base_positions_deg[jn]
            # Hold gripper
            if "gripper" in base_positions_deg:
                cmd["gripper.pos"] = base_positions_deg["gripper"]

            robot.send_action(cmd)
            obs = robot.get_observation()

            timestamps.append(t)
            positions.append([obs.get(f"{jn}.pos", 0.0) for jn in joint_names])
            commanded.append([cmd[f"{jn}.pos"] for jn in joint_names])

            sleep_time = config.dt - (time.monotonic() - t0 - t)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Trial interrupted")
    finally:
        # Return to base
        cmd = {f"{jn}.pos": base_positions_deg[jn] for jn in joint_names}
        if "gripper" in base_positions_deg:
            cmd["gripper.pos"] = base_positions_deg["gripper"]
        robot.send_action(cmd)

    logger.info(f"Trial complete: {len(timestamps)} samples in {t:.1f}s")
    return {
        "timestamps_s": timestamps,
        "positions_deg": positions,
        "commanded_deg": commanded,
        "n_samples": len(timestamps),
    }


def run_config(robot, config_name, q_rad, args):
    """Run perturbation trials for all joints at one configuration."""
    ir = make_piper_single_arm_ir()
    q_arr = np.array(q_rad)
    theory = compute_theoretical_coupling(ir, q_arr)
    J = np.array(theory["J_matrix"])

    q_deg = np.degrees(q_arr)
    base_positions = {f"joint_{i+1}": float(q_deg[i]) for i in range(6)}

    # Read current gripper position to hold it
    obs = robot.get_observation()
    base_positions["gripper"] = obs.get("gripper.pos", 0.0)

    logger.info(f"=== Config: {config_name} ===")
    logger.info(f"  q_rad = {q_rad}")
    logger.info(f"  q_deg = {q_deg.tolist()}")

    # Slow move to target configuration
    target = {f"{k}.pos": v for k, v in base_positions.items()}
    logger.info("  Moving to configuration...")
    slow_move(robot, target, duration_s=3.0)
    time.sleep(1.0)

    pconfig = PerturbationConfig(
        amplitude_deg=args.amplitude,
        frequency_hz=args.frequency,
        duration_s=args.duration,
        ramp_s=2.0,
    )

    for j_idx in range(args.joint_start, args.joint_end + 1):
        jn = f"joint_{j_idx + 1}"
        logger.info(f"  Perturbing {jn} (idx={j_idx})...")

        trial = run_position_coupling_trial(
            robot, JOINT_NAMES, base_positions, jn, pconfig,
        )

        # Predicted strongest response
        j_row = np.abs(J[j_idx])
        j_row[j_idx] = 0
        pred_strongest = JOINT_NAMES[np.argmax(j_row)]
        pred_J = j_row[np.argmax(j_row)]
        logger.info(f"    Predicted strongest: {pred_strongest} (|J|={pred_J:.3f})")

        result = {
            "experiment": "piper_position_coupling",
            "robot": "Piper_6DOF",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_name": config_name,
            "config_q_rad": q_rad,
            "perturbed_joint_idx": j_idx,
            "perturbed_joint_name": jn,
            "perturbation": {
                "amplitude_deg": pconfig.amplitude_deg,
                "frequency_hz": pconfig.frequency_hz,
                "duration_s": pconfig.duration_s,
                "ramp_s": pconfig.ramp_s,
            },
            "joint_names": JOINT_NAMES,
            "data_type": "position_only",
            **trial,
            "theoretical": theory,
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"piper_coupling_{config_name}_j{j_idx}.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"    Saved: {fname}")
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Piper 6-DOF coupling experiment")
    parser.add_argument("--port", default="can2",
                        help="Piper follower CAN port")
    parser.add_argument("--speed-ratio", type=int, default=35,
                        help="Piper motion speed (0-100)")
    parser.add_argument("--config", default="home",
                        help="Config name or 'all'")
    parser.add_argument("--joint-start", type=int, default=0)
    parser.add_argument("--joint-end", type=int, default=5)
    parser.add_argument("--amplitude", type=float, default=3.0,
                        help="Perturbation amplitude (deg)")
    parser.add_argument("--frequency", type=float, default=0.5,
                        help="Perturbation frequency (Hz)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Trial duration (s)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute theory only, no robot")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: theoretical predictions only")
        ir = make_piper_single_arm_ir()
        configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
        for name, q in configs.items():
            theory = compute_theoretical_coupling(ir, np.array(q))
            J = np.array(theory["J_matrix"])
            logger.info(f"\n{name}: q = {q}")
            logger.info(f"  Top-5 |J_ij|:")
            pairs = []
            for i in range(6):
                for j in range(i + 1, 6):
                    pairs.append((abs(J[i, j]), i, j))
            pairs.sort(reverse=True)
            for val, i, j in pairs[:5]:
                logger.info(f"    ({i},{j}): |J|={val:.4f}")
        return

    from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig

    rconfig = PiperFollowerConfig(
        port=args.port,
        id="piper_coupling_exp",
        speed_ratio=args.speed_ratio,
        high_follow=False,
        require_calibration=False,
        startup_sleep_s=0.5,
        sync_gripper=True,
    )
    robot = PiperFollower(rconfig)

    try:
        robot.connect()
        logger.info(f"Connected to Piper on {args.port} (speed_ratio={args.speed_ratio})")

        configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
        for name, q in configs.items():
            run_config(robot, name, q, args)

        logger.info("All Piper trials complete!")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
