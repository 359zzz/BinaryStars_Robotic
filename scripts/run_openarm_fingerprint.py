#!/usr/bin/env python3
"""Experiment E: Dynamic coupling fingerprint — automated motion + recording.

Moves the robot through 4 predefined motions, recording q(t) at ~100Hz.
Offline analysis computes J_ij(q(t)) and generates coupling heatmap animation.

Usage:
    python scripts/run_openarm_fingerprint.py --port can0 --side right
    python analysis/fingerprint_animation.py results/fingerprint_trajectory.json
"""

import argparse
import json
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.experiment.safety import slow_move

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# Joint limits (deg): j1[-75,75] j2[-9,90] j3[-85,85] j4[0,135] j5[-85,85] j6[-40,40] j7[-80,80]

# 4 motions: each is a list of waypoints (q in degrees for 7 joints)
# Robot ramps between waypoints at ~3s each, recording continuously.
MOTIONS = {
    "shoulder_lift": {
        "description": "Lift shoulder from 0 to 80 degrees and back",
        "waypoints": [
            [0, 0, 0, 0, 0, 0, 0],       # home
            [0, 80, 0, 0, 0, 0, 0],       # shoulder up
            [0, 0, 0, 0, 0, 0, 0],        # back to home
        ],
        "segment_duration_s": 4.0,
    },
    "elbow_bend": {
        "description": "Bend elbow from 0 to 120 degrees and back",
        "waypoints": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 120, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        "segment_duration_s": 4.0,
    },
    "wrist_rotation": {
        "description": "Rotate wrist from -30 to +30 degrees",
        "waypoints": [
            [0, 0, 0, 0, 0, -30, 0],
            [0, 0, 0, 0, 0, +30, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        "segment_duration_s": 3.0,
    },
    "full_reach": {
        "description": "Combined shoulder + elbow + wrist reach motion",
        "waypoints": [
            [0, 0, 0, 0, 0, 0, 0],           # home
            [0, 40, 0, 60, 0, 0, 0],          # mid reach
            [0, 80, 0, 120, 0, 30, 0],        # full reach
            [30, 80, 0, 120, 0, 30, 0],       # rotate base
            [0, 0, 0, 0, 0, 0, 0],            # home
        ],
        "segment_duration_s": 4.0,
    },
}


def execute_motion_with_recording(robot, motion_name, motion_def, dt=0.02):
    """Execute motion waypoints, recording q(t) continuously."""
    waypoints = motion_def["waypoints"]
    seg_dur = motion_def["segment_duration_s"]

    logger.info(f"  Motion: {motion_name} — {motion_def['description']}")
    logger.info(f"  Waypoints: {len(waypoints)}, segment duration: {seg_dur}s")

    timestamps = []
    positions_deg = []
    torques_Nm = []

    # Move to first waypoint
    wp0 = waypoints[0]
    target = {}
    for i, jn in enumerate(JOINT_NAMES):
        target[f"{jn}.pos"] = wp0[i]
    obs = robot.get_observation()
    target["gripper.pos"] = obs.get("gripper.pos", 0.0)
    slow_move(robot, target, duration_s=3.0)
    time.sleep(0.5)

    gripper_pos = obs.get("gripper.pos", 0.0)
    t_global = 0.0

    for seg_idx in range(len(waypoints) - 1):
        wp_start = waypoints[seg_idx]
        wp_end = waypoints[seg_idx + 1]
        n_steps = max(int(seg_dur / dt), 1)

        logger.info(f"  Segment {seg_idx}: {wp_start} → {wp_end}")

        t0 = time.monotonic()
        for step in range(n_steps + 1):
            alpha = step / n_steps
            cmd = {}
            for i, jn in enumerate(JOINT_NAMES):
                cmd[f"{jn}.pos"] = wp_start[i] + alpha * (wp_end[i] - wp_start[i])
            cmd["gripper.pos"] = gripper_pos

            robot.send_action(cmd)
            obs = robot.get_observation()

            t_now = time.monotonic() - t0 + t_global
            timestamps.append(t_now)
            positions_deg.append([obs.get(f"{jn}.pos", 0.0) for jn in JOINT_NAMES])
            torques_Nm.append([obs.get(f"{jn}.torque", 0.0) for jn in JOINT_NAMES])

            elapsed = time.monotonic() - t0
            target_t = step * dt
            if target_t > elapsed:
                time.sleep(target_t - elapsed)

        t_global += seg_dur

    logger.info(f"  Recorded {len(timestamps)} samples over {t_global:.1f}s")

    return {
        "motion_name": motion_name,
        "description": motion_def["description"],
        "waypoints": waypoints,
        "segment_duration_s": seg_dur,
        "timestamps_s": timestamps,
        "positions_deg": positions_deg,
        "torques_Nm": torques_Nm,
        "n_samples": len(timestamps),
    }


def main():
    parser = argparse.ArgumentParser(description="OpenArm dynamic coupling fingerprint")
    parser.add_argument("--port", default="can0")
    parser.add_argument("--side", default="right")
    parser.add_argument("--motions", nargs="+", default=list(MOTIONS.keys()),
                        help="Which motions to run")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    rconfig = OpenArmFollowerConfig(port=args.port, side=args.side, id="fingerprint_exp")
    robot = OpenArmFollower(rconfig)

    try:
        robot.connect()
        logger.info("Connected to OpenArm")

        all_motions = []
        for name in args.motions:
            if name not in MOTIONS:
                logger.warning(f"Unknown motion: {name}, skipping")
                continue
            logger.info(f"\n=== Motion: {name} ===")
            result = execute_motion_with_recording(robot, name, MOTIONS[name])
            all_motions.append(result)
            time.sleep(1.0)

        # Return to home
        logger.info("Returning to home...")
        home = {f"{jn}.pos": 0.0 for jn in JOINT_NAMES}
        obs = robot.get_observation()
        home["gripper.pos"] = obs.get("gripper.pos", 0.0)
        slow_move(robot, home, duration_s=3.0)

        # Save
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "experiment": "dynamic_fingerprint",
            "robot": "OpenArm_v10",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "joint_names": JOINT_NAMES,
            "motions": all_motions,
        }
        fname = out_dir / "fingerprint_trajectory.json"
        with open(fname, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nSaved: {fname}")
        logger.info("Run offline analysis: python analysis/fingerprint_animation.py results/fingerprint_trajectory.json")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
