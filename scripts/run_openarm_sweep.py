#!/usr/bin/env python3
"""Experiment 3: Elbow configuration sweep — track coupling landscape change.

Sweeps q4 (elbow) through a range, perturbs base joint at each step,
records torque response. Shows coupling graph evolves with configuration.
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

from bsreal.robot_data.openarm_data import make_openarm_single_arm_ir
from bsreal.experiment.perturbation import (
    PerturbationConfig,
    run_perturbation_trial,
    compute_theoretical_coupling,
)
from bsreal.experiment.safety import slow_move

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def main():
    parser = argparse.ArgumentParser(description="OpenArm elbow configuration sweep")
    parser.add_argument("--port", default="can0")
    parser.add_argument("--side", default="right")
    parser.add_argument("--sweep-min", type=float, default=-60.0, help="Elbow min (deg)")
    parser.add_argument("--sweep-max", type=float, default=60.0, help="Elbow max (deg)")
    parser.add_argument("--sweep-steps", type=int, default=21)
    parser.add_argument("--perturb-joint", type=int, default=0, help="Joint to perturb (idx)")
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=5.0, help="Shorter per step")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ir = make_openarm_single_arm_ir()
    sweep_deg = np.linspace(args.sweep_min, args.sweep_max, args.sweep_steps)
    sweep_rad = np.radians(sweep_deg)

    if args.dry_run:
        logger.info("DRY RUN: theoretical sweep")
        for q4d, q4r in zip(sweep_deg, sweep_rad):
            q = np.array([0, 0, 0, q4r, 0, 0, 0])
            th = compute_theoretical_coupling(ir, q)
            J = np.array(th["J_matrix"])
            j_idx = args.perturb_joint
            coupling = [abs(J[j_idx, k]) for k in range(7) if k != j_idx]
            logger.info(f"  q4={q4d:+6.1f} deg: max|J(0,k)|={max(coupling):.3f}")
        return

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    rconfig = OpenArmFollowerConfig(port=args.port, side=args.side, id="sweep_exp")
    robot = OpenArmFollower(rconfig)

    pconfig = PerturbationConfig(
        amplitude_deg=args.amplitude,
        frequency_hz=args.frequency,
        duration_s=args.duration,
        ramp_s=1.5,
    )

    all_steps = []

    try:
        robot.connect()
        logger.info(f"Connected. Sweeping q4 from {args.sweep_min} to {args.sweep_max} deg")

        for step_i, (q4d, q4r) in enumerate(zip(sweep_deg, sweep_rad)):
            q_rad = np.array([0, 0, 0, q4r, 0, 0, 0])
            q_deg = np.degrees(q_rad)
            base_pos = {f"joint_{i+1}": float(q_deg[i]) for i in range(7)}

            obs = robot.get_observation()
            base_pos["gripper"] = obs.get("gripper.pos", 0.0)

            logger.info(f"[{step_i+1}/{len(sweep_deg)}] q4 = {q4d:+.1f} deg")

            target = {f"{k}.pos": v for k, v in base_pos.items()}
            slow_move(robot, target, duration_s=2.0)
            time.sleep(0.5)

            jn = f"joint_{args.perturb_joint + 1}"
            trial = run_perturbation_trial(
                robot, JOINT_NAMES, base_pos, jn, pconfig
            )

            theory = compute_theoretical_coupling(ir, q_rad)

            all_steps.append({
                "step": step_i,
                "q4_deg": float(q4d),
                "q4_rad": float(q4r),
                "n_samples": len(trial.timestamps_s),
                "timestamps_s": trial.timestamps_s,
                "positions_deg": trial.positions_deg,
                "torques_Nm": trial.torques_Nm,
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

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "elbow_sweep",
        "robot": "OpenArm_v10",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "perturb_joint_idx": args.perturb_joint,
        "sweep_joint_idx": 3,
        "sweep_range_deg": [args.sweep_min, args.sweep_max],
        "n_steps": len(all_steps),
        "steps": all_steps,
    }
    fname = out_dir / "openarm_elbow_sweep.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {fname}")


if __name__ == "__main__":
    main()
