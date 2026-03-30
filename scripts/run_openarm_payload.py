#!/usr/bin/env python3
"""Experiment C: Payload-dependent coupling shift (OpenArm 7-DOF).

Run the same perturbation protocol at one configuration with different
end-effector payloads (0, 0.5, 1.0 kg).  The quantum model predicts
the coupling change from a single parameter update (Δm₇).

Usage:
    # 0 kg (no payload) — can reuse existing data
    python scripts/run_openarm_payload.py --port can0 --side right --payload 0.0

    # Attach 0.5 kg, then:
    python scripts/run_openarm_payload.py --port can0 --side right --payload 0.5

    # Attach 1.0 kg, then:
    python scripts/run_openarm_payload.py --port can0 --side right --payload 1.0
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

from bsreal.robot_data.openarm_data import (
    OPENARM_LINK_MASSES, OPENARM_LINK_INERTIAS, OPENARM_LINK_COMS,
    OPENARM_JOINT_AXES, OPENARM_JOINT_XYZ,
    make_openarm_single_arm_ir,
)
from bsreal.dynamics.models import DynamicsIR
from bsreal.dynamics.mass_matrix import compute_mass_matrix
from bsreal.dynamics.coupling import normalized_coupling_matrix
from bsreal.experiment.perturbation import (
    PerturbationConfig,
    run_perturbation_trial,
)
from bsreal.experiment.safety import slow_move

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# Use elbow_up config — non-trivial coupling landscape
CONFIG_Q_RAD = [0.0, 0.0, 0.0, +1.2, 0.0, 0.0, 0.0]
CONFIG_NAME = "elbow_up"


def make_ir_with_payload(payload_kg: float):
    """Create OpenArm IR with extra mass on link 7 (end-effector)."""
    ir_base = make_openarm_single_arm_ir()
    if payload_kg <= 0:
        return ir_base

    # Add payload to last link's mass
    new_masses = list(ir_base.link_masses)
    new_masses[6] = float(new_masses[6]) + payload_kg

    return DynamicsIR(
        name=f"openarm_v10_payload_{payload_kg}kg",
        topology=ir_base.topology,
        parent_indices=ir_base.parent_indices,
        tree_depth=ir_base.tree_depth,
        mass_matrix_bandwidth=ir_base.mass_matrix_bandwidth,
        n_joints=ir_base.n_joints,
        joint_names=ir_base.joint_names,
        joint_axes_local=ir_base.joint_axes_local,
        parent_to_joint_transforms=ir_base.parent_to_joint_transforms,
        base_transform=ir_base.base_transform,
        link_masses=tuple(new_masses),
        link_inertias=ir_base.link_inertias,
        link_com_local=ir_base.link_com_local,
        gravity=ir_base.gravity,
    )


def compute_theory(payload_kg, q_rad):
    ir = make_ir_with_payload(payload_kg)
    q = np.array(q_rad)
    M = compute_mass_matrix(ir, q)
    J = normalized_coupling_matrix(M)
    return {"M_matrix": M.tolist(), "J_matrix": J.tolist(), "payload_kg": payload_kg}


def main():
    parser = argparse.ArgumentParser(description="OpenArm payload coupling experiment")
    parser.add_argument("--port", default="can0")
    parser.add_argument("--side", default="right")
    parser.add_argument("--payload", type=float, required=True,
                        help="Payload mass in kg (0, 0.5, or 1.0)")
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload_str = f"{args.payload:.1f}".replace(".", "p")
    q_arr = np.array(CONFIG_Q_RAD)

    # Theory for 0 kg and current payload
    theory_0 = compute_theory(0.0, CONFIG_Q_RAD)
    theory_p = compute_theory(args.payload, CONFIG_Q_RAD)

    J_0 = np.array(theory_0["J_matrix"])
    J_p = np.array(theory_p["J_matrix"])
    J_diff = np.abs(J_p) - np.abs(J_0)

    logger.info(f"Payload: {args.payload} kg")
    logger.info(f"Config: {CONFIG_NAME} = {CONFIG_Q_RAD}")
    logger.info(f"\nTop-5 coupling CHANGES (|J_payload| - |J_0|):")
    pairs = []
    for i in range(7):
        for j in range(i + 1, 7):
            pairs.append((abs(J_diff[i, j]), J_0[i, j], J_p[i, j], i, j))
    pairs.sort(reverse=True)
    for diff, j0, jp, i, j in pairs[:5]:
        logger.info(f"  ({i},{j}): J_0={j0:+.4f} → J_p={jp:+.4f} (Δ={diff:+.4f})")

    if args.dry_run:
        logger.info("DRY RUN complete.")
        return

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    rconfig = OpenArmFollowerConfig(port=args.port, side=args.side, id="payload_exp")
    robot = OpenArmFollower(rconfig)

    try:
        robot.connect()
        logger.info(f"Connected to OpenArm on {args.port}")

        q_deg = np.degrees(q_arr)
        base_positions = {f"joint_{i+1}": float(q_deg[i]) for i in range(7)}
        obs = robot.get_observation()
        base_positions["gripper"] = obs.get("gripper.pos", 0.0)

        target = {f"{k}.pos": v for k, v in base_positions.items()}
        logger.info("Moving to configuration...")
        slow_move(robot, target, duration_s=3.0)
        time.sleep(1.0)

        pconfig = PerturbationConfig(
            amplitude_deg=args.amplitude,
            frequency_hz=args.frequency,
            duration_s=args.duration,
            ramp_s=2.0,
        )

        for j_idx in range(7):
            jn = f"joint_{j_idx + 1}"
            logger.info(f"Perturbing {jn} (payload={args.payload} kg)...")

            trial = run_perturbation_trial(
                robot=robot, joint_names=JOINT_NAMES,
                base_positions_deg=base_positions,
                perturb_joint=jn, config=pconfig,
            )

            result = {
                "experiment": "payload_coupling",
                "robot": "OpenArm_v10",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_name": CONFIG_NAME,
                "config_q_rad": CONFIG_Q_RAD,
                "payload_kg": args.payload,
                "perturbed_joint_idx": j_idx,
                "perturbed_joint_name": jn,
                "perturbation": {
                    "amplitude_deg": pconfig.amplitude_deg,
                    "frequency_hz": pconfig.frequency_hz,
                    "duration_s": pconfig.duration_s,
                    "ramp_s": pconfig.ramp_s,
                },
                "joint_names": JOINT_NAMES,
                "n_samples": len(trial.timestamps_s),
                "timestamps_s": trial.timestamps_s,
                "positions_deg": trial.positions_deg,
                "velocities_deg_s": trial.velocities_deg_s,
                "torques_Nm": trial.torques_Nm,
                "commanded_deg": trial.commanded_deg,
                "theoretical_0kg": theory_0,
                "theoretical_payload": theory_p,
            }

            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"payload_{payload_str}kg_j{j_idx}.json"
            with open(fname, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"  Saved: {fname}")
            time.sleep(1.0)

        logger.info("Payload experiment complete!")
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
