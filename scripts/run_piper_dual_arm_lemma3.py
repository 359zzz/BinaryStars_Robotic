#!/usr/bin/env python3
"""Piper dual-arm Lemma 3 verification.

Perturbs one arm's joint and verifies zero coupling response on the other arm.
Uses BiPiperFollower (12-DOF dual-arm) with position perturbation
(Piper has no torque readout — uses position deviation as proxy).
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

from bsreal.experiment.perturbation import PerturbationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RIGHT_JOINTS = [f"right_joint_{i}" for i in range(1, 7)]
LEFT_JOINTS = [f"left_joint_{i}" for i in range(1, 7)]
ALL_JOINTS = RIGHT_JOINTS + LEFT_JOINTS


def run_cross_arm_trial(robot, perturb_arm: str, perturb_joint_idx: int, args):
    """Perturb one arm, record all 12 joint positions."""
    if perturb_arm == "right":
        perturb_jn = f"right_joint_{perturb_joint_idx + 1}"
        own_joints = RIGHT_JOINTS
        other_joints = LEFT_JOINTS
    else:
        perturb_jn = f"left_joint_{perturb_joint_idx + 1}"
        own_joints = LEFT_JOINTS
        other_joints = RIGHT_JOINTS

    pconfig = PerturbationConfig(
        amplitude_deg=args.amplitude,
        frequency_hz=args.frequency,
        duration_s=args.duration,
        ramp_s=2.0,
    )

    obs = robot.get_observation()
    base_positions = {}
    for jn in ALL_JOINTS:
        base_positions[jn] = obs.get(f"{jn}.pos", 0.0)
    # Include grippers
    base_positions["right_gripper.pos"] = obs.get("right_gripper.pos", 0.0)
    base_positions["left_gripper.pos"] = obs.get("left_gripper.pos", 0.0)

    logger.info(f"Perturbing {perturb_jn}, A={pconfig.amplitude_deg} deg")

    omega = 2 * math.pi * pconfig.frequency_hz
    timestamps = []
    own_positions = []
    other_positions = []
    own_deviations = []
    other_deviations = []

    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            if t >= pconfig.duration_s:
                break

            amp = pconfig.amplitude_deg * min(t / pconfig.ramp_s, 1.0)

            cmd = {}
            for jn in ALL_JOINTS:
                if jn == perturb_jn:
                    cmd[f"{jn}.pos"] = base_positions[jn] + amp * math.sin(omega * t)
                else:
                    cmd[f"{jn}.pos"] = base_positions[jn]
            cmd["right_gripper.pos"] = base_positions["right_gripper.pos"]
            cmd["left_gripper.pos"] = base_positions["left_gripper.pos"]

            robot.send_action(cmd)
            obs = robot.get_observation()

            timestamps.append(t)
            own_pos = [obs.get(f"{jn}.pos", 0.0) for jn in own_joints]
            other_pos = [obs.get(f"{jn}.pos", 0.0) for jn in other_joints]
            own_positions.append(own_pos)
            other_positions.append(other_pos)

            # Deviations from base
            own_dev = [obs.get(f"{jn}.pos", 0.0) - base_positions[jn] for jn in own_joints]
            other_dev = [obs.get(f"{jn}.pos", 0.0) - base_positions[jn] for jn in other_joints]
            own_deviations.append(own_dev)
            other_deviations.append(other_dev)

            sleep_time = pconfig.dt - (time.monotonic() - t0 - t)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cmd = {f"{jn}.pos": base_positions[jn] for jn in ALL_JOINTS}
        cmd["right_gripper.pos"] = base_positions["right_gripper.pos"]
        cmd["left_gripper.pos"] = base_positions["left_gripper.pos"]
        robot.send_action(cmd)

    # Compute cross-arm coupling metric
    other_dev_arr = np.array(other_deviations)
    own_dev_arr = np.array(own_deviations)
    other_max_dev = float(np.max(np.abs(other_dev_arr)))
    own_max_dev = float(np.max(np.abs(own_dev_arr)))

    logger.info(
        f"  own_max_dev={own_max_dev:.3f}  other_max_dev={other_max_dev:.3f}  "
        f"ratio={other_max_dev / max(own_max_dev, 1e-6):.4f}"
    )

    return {
        "perturb_arm": perturb_arm,
        "perturb_joint_idx": perturb_joint_idx,
        "perturb_joint_name": perturb_jn,
        "n_samples": len(timestamps),
        "timestamps_s": timestamps,
        "own_arm_joints": own_joints,
        "other_arm_joints": other_joints,
        "own_arm_positions": own_positions,
        "other_arm_positions": other_positions,
        "own_max_deviation_deg": own_max_dev,
        "other_max_deviation_deg": other_max_dev,
        "coupling_ratio": other_max_dev / max(own_max_dev, 1e-6),
        "perturbation": {
            "amplitude_deg": pconfig.amplitude_deg,
            "frequency_hz": pconfig.frequency_hz,
        },
    }


def run_theory_predictions():
    """Compute theoretical coupling for Piper dual-arm at q=0."""
    from bsreal.robot_data.piper_data import make_piper_dual_arm_ir
    from bsreal.dynamics.mass_matrix import compute_mass_matrix
    from bsreal.dynamics.coupling import normalized_coupling_matrix

    ir = make_piper_dual_arm_ir()
    q = np.zeros(12)
    M = compute_mass_matrix(ir, q)
    J = normalized_coupling_matrix(M)

    cross_block = J[:6, 6:]
    cross_max = float(np.max(np.abs(cross_block)))
    logger.info(f"Theory: cross-arm max|J_ij| = {cross_max:.6f}")
    logger.info(f"  (Expected: 0.0 for tree topology, verifies Lemma 3)")

    return {
        "M_matrix": M.tolist(),
        "J_matrix": J.tolist(),
        "cross_arm_max_J": cross_max,
        "lemma3_verified": cross_max < 1e-10,
    }


def main():
    parser = argparse.ArgumentParser(description="Piper dual-arm Lemma 3 verification")
    parser.add_argument("--left-port", default="can1")
    parser.add_argument("--right-port", default="can0")
    parser.add_argument("--amplitude", type=float, default=5.0,
                        help="Perturbation amplitude (deg, larger for Piper)")
    parser.add_argument("--frequency", type=float, default=0.3,
                        help="Perturbation frequency (Hz, lower for Piper)")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--joints", nargs="+", type=int, default=[0, 2, 4],
                        help="Joint indices to perturb (0-5)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--theory-only", action="store_true",
                        help="Only compute theory predictions, skip hardware")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theory = run_theory_predictions()

    if args.theory_only:
        result = {
            "experiment": "piper_dual_arm_lemma3",
            "robot": "Piper_dual",
            "theory_only": True,
            "theory": theory,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        fname = out_dir / "piper_dual_arm_lemma3.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved: {fname}")
        return

    from lerobot.robots.bi_piper_follower import (
        BiPiperFollower, BiPiperFollowerConfig,
    )
    from lerobot.robots.piper_follower import PiperFollowerConfig

    config = BiPiperFollowerConfig(
        left_arm_config=PiperFollowerConfig(port=args.left_port, id="left"),
        right_arm_config=PiperFollowerConfig(port=args.right_port, id="right"),
        id="piper_lemma3",
    )
    robot = BiPiperFollower(config)

    try:
        robot.connect()
        logger.info("Connected to dual-arm Piper")

        all_trials = []
        for j_idx in args.joints:
            for arm in ["right", "left"]:
                logger.info(f"=== Perturb {arm} joint {j_idx} ===")
                trial = run_cross_arm_trial(robot, arm, j_idx, args)
                all_trials.append(trial)
                time.sleep(1.0)

        result = {
            "experiment": "piper_dual_arm_lemma3",
            "robot": "Piper_dual",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "theory": theory,
            "trials": all_trials,
        }
        fname = out_dir / "piper_dual_arm_lemma3.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved: {fname}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
