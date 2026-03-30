#!/usr/bin/env python3
"""Experiment 5: Piper elbow configuration sweep.

Sweeps joint 3 (elbow) through a range, perturbs base joint at each step,
records position response.  Position-only (Piper has no torque readout).
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


def run_sweep_trial(robot, joint_names, base_positions_deg, perturb_joint, config):
    """Single sweep-step perturbation trial (position only)."""
    omega = 2.0 * math.pi * config.frequency_hz
    base_val = base_positions_deg[perturb_joint]

    timestamps = []
    positions = []

    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            if t >= config.duration_s:
                break

            amp = config.amplitude_deg * min(t / config.ramp_s, 1.0)

            cmd = {}
            for jn in joint_names:
                if jn == perturb_joint:
                    cmd[f"{jn}.pos"] = base_val + amp * math.sin(omega * t)
                else:
                    cmd[f"{jn}.pos"] = base_positions_deg[jn]
            if "gripper" in base_positions_deg:
                cmd["gripper.pos"] = base_positions_deg["gripper"]

            robot.send_action(cmd)
            obs = robot.get_observation()

            timestamps.append(t)
            positions.append([obs.get(f"{jn}.pos", 0.0) for jn in joint_names])

            sleep_time = config.dt - (time.monotonic() - t0 - t)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cmd = {f"{jn}.pos": base_positions_deg[jn] for jn in joint_names}
        if "gripper" in base_positions_deg:
            cmd["gripper.pos"] = base_positions_deg["gripper"]
        robot.send_action(cmd)

    return timestamps, positions


def main():
    parser = argparse.ArgumentParser(description="Piper elbow configuration sweep")
    parser.add_argument("--port", default="can2")
    parser.add_argument("--speed-ratio", type=int, default=35)
    parser.add_argument("--sweep-min", type=float, default=-50.0, help="Elbow min (deg)")
    parser.add_argument("--sweep-max", type=float, default=10.0, help="Elbow max (deg)")
    parser.add_argument("--sweep-steps", type=int, default=13)
    parser.add_argument("--perturb-joint", type=int, default=0, help="Joint to perturb (idx)")
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ir = make_piper_single_arm_ir()
    sweep_deg = np.linspace(args.sweep_min, args.sweep_max, args.sweep_steps)
    sweep_rad = np.radians(sweep_deg)

    # Piper elbow is joint 3 (index 2)
    sweep_joint_idx = 2

    if args.dry_run:
        logger.info("DRY RUN: theoretical sweep")
        for q3d, q3r in zip(sweep_deg, sweep_rad):
            q = np.zeros(6)
            q[sweep_joint_idx] = q3r
            th = compute_theoretical_coupling(ir, q)
            J = np.array(th["J_matrix"])
            j_idx = args.perturb_joint
            coupling = [abs(J[j_idx, k]) for k in range(6) if k != j_idx]
            logger.info(f"  q3={q3d:+6.1f} deg: max|J({j_idx},k)|={max(coupling):.3f}")
        return

    from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig

    rconfig = PiperFollowerConfig(
        port=args.port,
        id="piper_sweep_exp",
        speed_ratio=args.speed_ratio,
        high_follow=False,
        require_calibration=False,
        startup_sleep_s=0.5,
        sync_gripper=True,
    )
    robot = PiperFollower(rconfig)

    pconfig = PerturbationConfig(
        amplitude_deg=args.amplitude,
        frequency_hz=args.frequency,
        duration_s=args.duration,
        ramp_s=1.5,
    )

    all_steps = []

    try:
        robot.connect()
        logger.info(f"Connected. Sweeping q3 from {args.sweep_min} to {args.sweep_max} deg")

        for step_i, (q3d, q3r) in enumerate(zip(sweep_deg, sweep_rad)):
            q_rad = np.zeros(6)
            q_rad[sweep_joint_idx] = q3r
            q_deg = np.degrees(q_rad)
            base_pos = {f"joint_{i+1}": float(q_deg[i]) for i in range(6)}

            obs = robot.get_observation()
            base_pos["gripper"] = obs.get("gripper.pos", 0.0)

            logger.info(f"[{step_i+1}/{len(sweep_deg)}] q3 = {q3d:+.1f} deg")

            target = {f"{k}.pos": v for k, v in base_pos.items()}
            slow_move(robot, target, duration_s=2.0)
            time.sleep(0.5)

            jn = f"joint_{args.perturb_joint + 1}"
            timestamps, positions = run_sweep_trial(
                robot, JOINT_NAMES, base_pos, jn, pconfig,
            )

            theory = compute_theoretical_coupling(ir, q_rad)

            all_steps.append({
                "step": step_i,
                "q3_deg": float(q3d),
                "q3_rad": float(q3r),
                "n_samples": len(timestamps),
                "timestamps_s": timestamps,
                "positions_deg": positions,
                "theoretical": theory,
            })

        logger.info("Sweep complete!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "piper_elbow_sweep",
        "robot": "Piper_6DOF",
        "data_type": "position_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "perturb_joint_idx": args.perturb_joint,
        "sweep_joint_idx": sweep_joint_idx,
        "sweep_range_deg": [args.sweep_min, args.sweep_max],
        "n_steps": len(all_steps),
        "steps": all_steps,
    }
    fname = out_dir / "piper_elbow_sweep.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {fname}")


if __name__ == "__main__":
    main()
