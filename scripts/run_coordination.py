#!/usr/bin/env python3
"""Run dual-arm coordination experiments (Tables 9-10, Fig 7).

Bar-holding protocol:
  Both arms extend forward, grippers hold a rigid aluminum bar.
  Trajectory: sagittal-plane oscillation (forward/backward).
  3 configs x 3 tasks x 4 controllers x 3 reps = 108 trials.

Usage:
    # Single trial (no bar, validation)
    python scripts/run_coordination.py --robot openarm \
        --task independent --controller decoupled --config bar_mid \
        --reps 1 --right-port can1 --left-port can2

    # Single trial (with bar)
    python scripts/run_coordination.py --robot openarm \
        --task bar_only --controller c_coupled --config bar_mid \
        --reps 1 --right-port can1 --left-port can2

    # Full suite
    python scripts/run_coordination.py --robot openarm --all \
        --right-port can1 --left-port can2

    # Dry run
    python scripts/run_coordination.py --robot openarm --dry-run
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.dynamics.effective_mass import make_object_spatial_inertia
from bsreal.experiment.controllers import make_controller
from bsreal.experiment.trajectory import TASK_OBJECTS, COORDINATION_CONFIGS
from bsreal.experiment.coordination import (
    CoordinationConfig,
    run_coordination_trial,
    run_coordination_suite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_robot_and_ir(args):
    """Create robot connection and DynamicsIR based on --robot flag."""
    if args.robot == "openarm":
        from bsreal.robot_data.openarm_data import make_openarm_dual_arm_ir
        ir = make_openarm_dual_arm_ir()
        n_per_arm = 7

        if not args.dry_run:
            from lerobot.robots.bi_openarm_follower import (
                BiOpenArmFollower, BiOpenArmFollowerConfig,
            )
            from lerobot.robots.openarm_follower.config_openarm_follower import (
                OpenArmFollowerConfigBase,
            )
            config = BiOpenArmFollowerConfig(
                left_arm_config=OpenArmFollowerConfigBase(port=args.left_port, side="left"),
                right_arm_config=OpenArmFollowerConfigBase(port=args.right_port, side="right"),
                id="coordination_exp",
            )
            robot = BiOpenArmFollower(config)
        else:
            robot = None

        return robot, ir, n_per_arm, "openarm"

    else:  # piper
        from bsreal.robot_data.piper_data import make_piper_dual_arm_ir
        ir = make_piper_dual_arm_ir()
        n_per_arm = 6

        if not args.dry_run:
            from lerobot.robots.bi_piper_follower import (
                BiPiperFollower, BiPiperFollowerConfig,
            )
            from lerobot.robots.piper_follower import PiperFollowerConfig
            config = BiPiperFollowerConfig(
                left_arm_config=PiperFollowerConfig(port=args.left_port, id="left"),
                right_arm_config=PiperFollowerConfig(port=args.right_port, id="right"),
                id="coordination_exp",
            )
            robot = BiPiperFollower(config)
        else:
            robot = None

        return robot, ir, n_per_arm, "piper"


def run_single(args):
    """Run a single task/controller/config combination."""
    robot, ir, n_per_arm, robot_type = make_robot_and_ir(args)

    obj = TASK_OBJECTS[args.task[0]]
    M_obj = None
    if obj["mass"] > 0:
        M_obj = make_object_spatial_inertia(obj["mass"], obj["geometry"], obj["dims"])

    ctrl = make_controller(
        args.controller[0], ir, n_per_arm, robot_type, M_obj=M_obj,
    )

    jn_right = [f"right_joint_{i}" for i in range(1, n_per_arm + 1)]
    jn_left = [f"left_joint_{i}" for i in range(1, n_per_arm + 1)]

    cc = CoordinationConfig(
        task_name=args.task[0],
        object_mass_kg=obj["mass"],
        object_geometry=obj["geometry"],
        object_dims=obj.get("dims", ()),
        controller_name=args.controller[0],
        config_name=args.config[0],
        n_repetitions=args.reps,
        duration_s=args.duration,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if robot and not args.dry_run:
        robot.connect()

    try:
        for rep in range(args.reps):
            result = run_coordination_trial(
                robot, ctrl, ir, cc, rep,
                jn_right, jn_left, dry_run=args.dry_run,
            )
            fname = out_dir / f"{cc.task_name}_{cc.controller_name}_{cc.config_name}_rep{rep}.json"
            data = {
                "task": cc.task_name, "controller": cc.controller_name,
                "config": cc.config_name, "rep": rep,
                "rmse_right": result.rmse_right, "rmse_left": result.rmse_left,
                "rmse_total": result.rmse_total,
                "s_rho_l": result.s_rho_l, "j_cross_max": result.j_cross_max,
                "n_samples": result.n_samples,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved: {fname}")

            if not args.dry_run:
                time.sleep(1.0)
    finally:
        if robot and not args.dry_run:
            robot.disconnect()


def run_suite(args):
    """Run the full coordination suite."""
    robot, ir, n_per_arm, robot_type = make_robot_and_ir(args)

    tasks = args.task if args.task else None
    controllers = args.controller if args.controller else None
    configs = args.config if args.config else None

    if robot and not args.dry_run:
        robot.connect()

    try:
        summary = run_coordination_suite(
            robot, ir, args.output_dir,
            n_per_arm=n_per_arm, robot_type=robot_type,
            tasks=tasks, controllers=controllers, configs=configs,
            n_reps=args.reps, dry_run=args.dry_run,
        )
        logger.info(f"Suite complete: {summary['n_trials']} trials")
    finally:
        if robot and not args.dry_run:
            robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Dual-arm coordination experiments")
    parser.add_argument("--robot", choices=["openarm", "piper"], default="openarm")
    parser.add_argument("--left-port", default="can2")
    parser.add_argument("--right-port", default="can1")
    parser.add_argument("--task", nargs="+",
                        choices=list(TASK_OBJECTS.keys()))
    parser.add_argument("--controller", nargs="+",
                        choices=["decoupled", "j_coupled", "c_coupled", "s_adaptive"])
    parser.add_argument("--config", nargs="+",
                        choices=list(COORDINATION_CONFIGS.keys()))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", default="results/coordination")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate trajectories without hardware")
    parser.add_argument("--all", action="store_true",
                        help="Run full suite (3 tasks x 4 controllers x 3 configs = 108 trials)")
    args = parser.parse_args()

    if args.all or (args.task is None and args.controller is None):
        run_suite(args)
    else:
        if not args.task or not args.controller or not args.config:
            parser.error("Specify --task, --controller, --config for single trial, or use --all")
        run_single(args)


if __name__ == "__main__":
    main()
