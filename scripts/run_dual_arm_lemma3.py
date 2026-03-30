#!/usr/bin/env python3
"""Experiment 2: Cross-arm zero coupling verification (Lemma 3).

Perturbs one arm and verifies zero torque response on the other arm.
Uses BiOpenArmFollower (14-DOF dual-arm).
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
from bsreal.experiment.safety import slow_move, emergency_freeze, SafetyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


RIGHT_JOINTS = [f"right_joint_{i}" for i in range(1, 8)]
LEFT_JOINTS = [f"left_joint_{i}" for i in range(1, 8)]
ALL_JOINTS = RIGHT_JOINTS + LEFT_JOINTS


def run_cross_arm_trial(robot, perturb_arm: str, perturb_joint_idx: int, args):
    """Perturb one arm's joint, record all 14 torques."""
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

    # Read current positions as base
    obs = robot.get_observation()
    base_positions = {}
    for jn in ALL_JOINTS:
        base_positions[jn] = obs.get(f"{jn}.pos", 0.0)

    logger.info(f"Perturbing {perturb_jn}, A={pconfig.amplitude_deg} deg")

    omega = 2 * math.pi * pconfig.frequency_hz
    timestamps = []
    own_torques = []
    other_torques = []

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

            robot.send_action(cmd)
            obs = robot.get_observation()

            timestamps.append(t)
            own_torques.append([obs.get(f"{jn}.torque", 0.0) for jn in own_joints])
            other_torques.append([obs.get(f"{jn}.torque", 0.0) for jn in other_joints])

            sleep_time = pconfig.dt - (time.monotonic() - t0 - t)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cmd = {f"{jn}.pos": base_positions[jn] for jn in ALL_JOINTS}
        robot.send_action(cmd)

    return {
        "perturb_arm": perturb_arm,
        "perturb_joint_idx": perturb_joint_idx,
        "perturb_joint_name": perturb_jn,
        "n_samples": len(timestamps),
        "timestamps_s": timestamps,
        "own_arm_joints": own_joints,
        "other_arm_joints": other_joints,
        "own_arm_torques_Nm": own_torques,
        "other_arm_torques_Nm": other_torques,
        "perturbation": {
            "amplitude_deg": pconfig.amplitude_deg,
            "frequency_hz": pconfig.frequency_hz,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Dual-arm Lemma 3 verification")
    parser.add_argument("--left-port", default="can1", help="Left arm CAN port")
    parser.add_argument("--right-port", default="can0", help="Right arm CAN port")
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--joints", nargs="+", type=int, default=[0, 3],
                        help="Which joints to perturb (indices 0-6)")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    from lerobot.robots.bi_openarm_follower import (
        BiOpenArmFollower, BiOpenArmFollowerConfig,
    )
    from lerobot.robots.openarm_follower.config_openarm_follower import (
        OpenArmFollowerConfigBase,
    )

    config = BiOpenArmFollowerConfig(
        left_arm_config=OpenArmFollowerConfigBase(port=args.left_port, side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port=args.right_port, side="right"),
        id="lemma3_exp",
    )
    robot = BiOpenArmFollower(config)

    try:
        robot.connect()
        logger.info("Connected to dual-arm OpenArm")

        all_results = []
        for j_idx in args.joints:
            for arm in ["right", "left"]:
                logger.info(f"=== Perturb {arm} joint {j_idx} ===")
                trial = run_cross_arm_trial(robot, arm, j_idx, args)
                all_results.append(trial)
                time.sleep(1.0)

        # Save
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "experiment": "cross_arm_lemma3",
            "robot": "OpenArm_v10_dual",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trials": all_results,
        }
        fname = out_dir / "dual_arm_lemma3.json"
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
