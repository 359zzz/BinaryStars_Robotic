#!/usr/bin/env python3
"""Pre-flight check for dual-arm coordination experiments.

Tests: dual-arm connection, 14-joint feedback, gripper open/close,
safety limits, small synchronized motion.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.experiment.trajectory import get_start_positions_deg, COORDINATION_CONFIGS


def preflight_openarm_dual(left_port: str, right_port: str):
    from lerobot.robots.bi_openarm_follower import (
        BiOpenArmFollower, BiOpenArmFollowerConfig,
    )
    from lerobot.robots.openarm_follower.config_openarm_follower import (
        OpenArmFollowerConfigBase,
    )

    config = BiOpenArmFollowerConfig(
        left_arm_config=OpenArmFollowerConfigBase(port=left_port, side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port=right_port, side="right"),
        id="preflight_coordination",
    )
    robot = BiOpenArmFollower(config)

    right_joints = [f"right_joint_{i}" for i in range(1, 8)]
    left_joints = [f"left_joint_{i}" for i in range(1, 8)]
    all_joints = right_joints + left_joints

    # Step 1: Connect
    print("[1/6] Connecting dual-arm OpenArm...")
    robot.connect()
    print("  OK: connected")

    # Step 2: Read all 14 joints
    print("[2/6] Reading 14 joint states...")
    obs = robot.get_observation()
    for jn in all_joints:
        pos = obs.get(f"{jn}.pos", float("nan"))
        vel = obs.get(f"{jn}.vel", float("nan"))
        tau = obs.get(f"{jn}.torque", float("nan"))
        print(f"  {jn:20s}: pos={pos:+7.2f}  vel={vel:+7.2f}  tau={tau:+6.3f}")

    missing = [jn for jn in all_joints if f"{jn}.pos" not in obs]
    if missing:
        print(f"  WARNING: missing feedback for {missing}")
    else:
        print("  OK: all 14 joints reporting")

    # Step 3: Gripper test
    print("[3/6] Gripper test (close then open)...")
    robot.send_action({"right_gripper.pos": 0.0, "left_gripper.pos": 0.0})
    time.sleep(1.0)
    obs_closed = robot.get_observation()
    rg = obs_closed.get("right_gripper.pos", float("nan"))
    lg = obs_closed.get("left_gripper.pos", float("nan"))
    print(f"  Closed: right={rg:.1f}  left={lg:.1f}")

    robot.send_action({"right_gripper.pos": 90.0, "left_gripper.pos": 90.0})
    time.sleep(1.0)
    obs_open = robot.get_observation()
    rg2 = obs_open.get("right_gripper.pos", float("nan"))
    lg2 = obs_open.get("left_gripper.pos", float("nan"))
    print(f"  Open:   right={rg2:.1f}  left={lg2:.1f}")

    if abs(rg2 - rg) > 10 and abs(lg2 - lg) > 10:
        print("  OK: both grippers responding")
    else:
        print("  WARNING: gripper response may be weak")

    # Step 4: Safety limits check
    print("[4/6] Safety limits check...")
    from bsreal.experiment.safety import check_within_limits
    try:
        cur_pos = {}
        for jn in right_joints:
            short = jn.replace("right_", "")
            cur_pos[short] = obs.get(f"{jn}.pos", 0.0)
        check_within_limits(cur_pos)
        print("  OK: right arm within limits")
    except Exception as e:
        print(f"  WARNING: right arm limit issue: {e}")

    try:
        cur_pos = {}
        for jn in left_joints:
            short = jn.replace("left_", "")
            cur_pos[short] = obs.get(f"{jn}.pos", 0.0)
        check_within_limits(cur_pos)
        print("  OK: left arm within limits")
    except Exception as e:
        print(f"  WARNING: left arm limit issue: {e}")

    # Step 5: Small synchronized motion
    print("[5/6] Synchronized micro-motion (joint_2 +2 deg both arms)...")
    base = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in all_joints}
    target = dict(base)
    target["right_joint_2.pos"] = base["right_joint_2.pos"] + 2.0
    target["left_joint_2.pos"] = base["left_joint_2.pos"] + 2.0
    robot.send_action(target)
    time.sleep(0.8)

    obs2 = robot.get_observation()
    dr = obs2.get("right_joint_2.pos", 0.0) - obs.get("right_joint_2.pos", 0.0)
    dl = obs2.get("left_joint_2.pos", 0.0) - obs.get("left_joint_2.pos", 0.0)
    print(f"  Right delta={dr:+.2f}  Left delta={dl:+.2f} deg")

    # Return
    robot.send_action(base)
    time.sleep(0.5)

    sync_error = abs(dr - dl)
    if sync_error < 1.0 and abs(dr) > 0.5:
        print(f"  OK: synchronized (sync error = {sync_error:.2f} deg)")
    else:
        print(f"  WARNING: sync error = {sync_error:.2f} deg")

    # Step 6: Trajectory preview
    print("[6/6] Trajectory preview (home config)...")
    start_pos = get_start_positions_deg("home")
    print(f"  Start positions: { {k: f'{v:.1f}' for k, v in list(start_pos.items())[:4]} } ...")
    print("  OK: trajectory module loaded")

    print("\n[6/6] Disconnecting...")
    robot.disconnect()
    print("  OK: pre-flight PASSED")


def preflight_piper_dual(left_port: str, right_port: str):
    from lerobot.robots.bi_piper_follower import (
        BiPiperFollower, BiPiperFollowerConfig,
    )
    from lerobot.robots.piper_follower import PiperFollowerConfig

    config = BiPiperFollowerConfig(
        left_arm_config=PiperFollowerConfig(port=left_port, id="left"),
        right_arm_config=PiperFollowerConfig(port=right_port, id="right"),
        id="preflight_coordination",
    )
    robot = BiPiperFollower(config)

    right_joints = [f"right_joint_{i}" for i in range(1, 7)]
    left_joints = [f"left_joint_{i}" for i in range(1, 7)]
    all_joints = right_joints + left_joints

    print("[1/4] Connecting dual-arm Piper...")
    robot.connect()
    print("  OK: connected")

    print("[2/4] Reading 12 joint states...")
    obs = robot.get_observation()
    for jn in all_joints:
        pos = obs.get(f"{jn}.pos", float("nan"))
        print(f"  {jn:20s}: pos={pos:+7.2f}")

    print("[3/4] Micro-motion test (joint_2 +3 deg)...")
    base = {f"{jn}.pos": obs.get(f"{jn}.pos", 0.0) for jn in all_joints}
    base["right_gripper.pos"] = obs.get("right_gripper.pos", 0.0)
    base["left_gripper.pos"] = obs.get("left_gripper.pos", 0.0)
    target = dict(base)
    target["right_joint_2.pos"] = base["right_joint_2.pos"] + 3.0
    target["left_joint_2.pos"] = base["left_joint_2.pos"] + 3.0
    for _ in range(150):
        robot.send_action(target)
        time.sleep(0.01)

    obs2 = robot.get_observation()
    dr = obs2.get("right_joint_2.pos", 0.0) - obs.get("right_joint_2.pos", 0.0)
    dl = obs2.get("left_joint_2.pos", 0.0) - obs.get("left_joint_2.pos", 0.0)
    print(f"  Right delta={dr:+.2f}  Left delta={dl:+.2f} deg")

    for _ in range(150):
        robot.send_action(base)
        time.sleep(0.01)

    if abs(dr) > 0.5 and abs(dl) > 0.5:
        print("  OK: both arms responding")
    else:
        print("  WARNING: weak response")

    print("[4/4] Disconnecting...")
    robot.disconnect()
    print("  OK: pre-flight PASSED")


def main():
    parser = argparse.ArgumentParser(description="Pre-flight for coordination experiments")
    parser.add_argument("--robot", choices=["openarm", "piper"], default="openarm")
    parser.add_argument("--left-port", default="can1")
    parser.add_argument("--right-port", default="can0")
    args = parser.parse_args()

    try:
        if args.robot == "openarm":
            preflight_openarm_dual(args.left_port, args.right_port)
        else:
            preflight_piper_dual(args.left_port, args.right_port)
    except Exception as e:
        print(f"\nPRE-FLIGHT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
