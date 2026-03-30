#!/usr/bin/env python3
"""Experiment 1: Single-arm torque coupling verification (OpenArm 7-DOF).

For each configuration, perturbs each joint with a sinusoidal position command
and records torque responses at all joints. Saves raw data + theoretical
predictions as JSON.
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

# Add parent dir so bsreal is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.robot_data.openarm_data import make_openarm_single_arm_ir
from bsreal.experiment.perturbation import (
    PerturbationConfig,
    run_perturbation_trial,
    compute_theoretical_coupling,
)
from bsreal.experiment.safety import slow_move

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Experiment configurations (name, q_rad for 7 joints)
# Joint limits (deg): j1[-75,75] j2[-9,90] j3[-85,85] j4[0,135] j5[-85,85] j6[-40,40] j7[-80,80]
CONFIGS = {
    "home":            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "elbow_up":        [0.0, 0.0, 0.0, +1.2, 0.0, 0.0, 0.0],         # q4=+68.8°
    "shoulder_elbow":  [0.0, +1.0, 0.0, +0.6, 0.0, 0.0, 0.0],        # q2=+57.3°, q4=+34.4°
    "full_pose":       [0.0, +0.8, 0.0, +1.0, 0.0, +0.5, 0.0],       # q2=+45.8°, q4=+57.3°, q6=+28.6°
    "wrist_twist":     [0.0, 0.0, 0.0, 0.0, 0.0, +0.6, 0.0],         # q6=+34.4°
}

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def run_config(robot, config_name: str, q_rad: list[float], args):
    """Run perturbation trials for all joints at one configuration."""
    ir = make_openarm_single_arm_ir()
    q_arr = np.array(q_rad)

    # Theoretical predictions
    theory = compute_theoretical_coupling(ir, q_arr)
    J = np.array(theory["J_matrix"])

    # Convert config to degrees for robot commands
    q_deg = np.degrees(q_arr)
    base_positions = {f"joint_{i+1}": float(q_deg[i]) for i in range(7)}

    # Add gripper (hold at current)
    obs = robot.get_observation()
    base_positions["gripper"] = obs.get("gripper.pos", 0.0)

    logger.info(f"=== Config: {config_name} ===")
    logger.info(f"  q_rad = {q_rad}")
    logger.info(f"  q_deg = {q_deg.tolist()}")

    # Move to configuration
    target_with_suffix = {f"{k}.pos": v for k, v in base_positions.items()}
    logger.info("  Moving to configuration...")
    slow_move(robot, target_with_suffix, duration_s=3.0)
    time.sleep(1.0)

    pconfig = PerturbationConfig(
        amplitude_deg=args.amplitude,
        frequency_hz=args.frequency,
        duration_s=args.duration,
        ramp_s=2.0,
    )

    joints_to_test = list(range(args.joint_start, args.joint_end + 1))

    for j_idx in joints_to_test:
        jn = f"joint_{j_idx + 1}"
        logger.info(f"  Perturbing {jn} (idx={j_idx})...")

        trial_data = run_perturbation_trial(
            robot=robot,
            joint_names=JOINT_NAMES,
            base_positions_deg=base_positions,
            perturb_joint=jn,
            config=pconfig,
        )

        # Predicted strongest response
        j_row = np.abs(J[j_idx])
        j_row[j_idx] = 0  # exclude self
        pred_strongest = JOINT_NAMES[np.argmax(j_row)]
        pred_J = j_row[np.argmax(j_row)]
        logger.info(f"    Predicted strongest: {pred_strongest} (|J|={pred_J:.3f})")

        # Save result
        result = {
            "experiment": "single_arm_coupling",
            "robot": "OpenArm_v10",
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
            "n_samples": len(trial_data.timestamps_s),
            "timestamps_s": trial_data.timestamps_s,
            "positions_deg": trial_data.positions_deg,
            "velocities_deg_s": trial_data.velocities_deg_s,
            "torques_Nm": trial_data.torques_Nm,
            "commanded_deg": trial_data.commanded_deg,
            "theoretical": theory,
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"openarm_coupling_{config_name}_j{j_idx}.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"    Saved: {fname}")

        # Brief pause between joints
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="OpenArm single-arm coupling experiment")
    parser.add_argument("--port", default="can0", help="CAN interface")
    parser.add_argument("--side", default="right", help="Arm side")
    parser.add_argument("--config", default="home", help="Config name or 'all'")
    parser.add_argument("--joint-start", type=int, default=0, help="First joint index (0-6)")
    parser.add_argument("--joint-end", type=int, default=6, help="Last joint index (0-6)")
    parser.add_argument("--amplitude", type=float, default=3.0, help="Perturbation amplitude (deg)")
    parser.add_argument("--frequency", type=float, default=0.5, help="Perturbation frequency (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Trial duration (s)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Compute theory only, no robot")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: computing theoretical predictions only")
        ir = make_openarm_single_arm_ir()
        configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
        for name, q in configs.items():
            theory = compute_theoretical_coupling(ir, np.array(q))
            J = np.array(theory["J_matrix"])
            logger.info(f"\n{name}: q = {q}")
            logger.info(f"  Top-5 |J_ij|:")
            pairs = []
            for i in range(7):
                for j in range(i + 1, 7):
                    pairs.append((abs(J[i, j]), i, j))
            pairs.sort(reverse=True)
            for val, i, j in pairs[:5]:
                logger.info(f"    ({i},{j}): |J|={val:.4f}")
        return

    # Connect to robot
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    rconfig = OpenArmFollowerConfig(port=args.port, side=args.side, id="coupling_exp")
    robot = OpenArmFollower(rconfig)

    try:
        robot.connect()
        logger.info(f"Connected to OpenArm on {args.port} ({args.side})")

        configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}

        for name, q in configs.items():
            run_config(robot, name, q, args)

        logger.info("All trials complete!")

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
