#!/usr/bin/env python3
"""Pre-flight check: connect to robot, read state, do micro-motion test."""

import argparse
import math
import sys
import time


def _openarm_gripper_targets(robot) -> tuple[float, float]:
    limits = robot.config.joint_limits.get("gripper", (-65.0, 0.0))
    return float(limits[0]), float(limits[1])


def _send_gripper_repeated(robot, target: float, duration_s: float = 0.8, dt: float = 0.05):
    cmd = {"gripper.pos": target}
    n_steps = max(int(duration_s / dt), 1)
    for _ in range(n_steps):
        robot.send_action(cmd)
        time.sleep(dt)
    robot.send_action(cmd)


def preflight_openarm(port: str, side: str = "right"):
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    config = OpenArmFollowerConfig(port=port, side=side, id="preflight_check")
    robot = OpenArmFollower(config)

    print(f"[1/4] Connecting to OpenArm on {port} ({side})...")
    robot.connect()
    print("  OK: connected")

    print("[2/4] Reading joint state...")
    obs = robot.get_observation()
    joint_names = [f"joint_{i}" for i in range(1, 8)]
    for jn in joint_names:
        pos = obs.get(f"{jn}.pos", float("nan"))
        vel = obs.get(f"{jn}.vel", float("nan"))
        tau = obs.get(f"{jn}.torque", float("nan"))
        print(f"  {jn}: pos={pos:+7.2f} deg  vel={vel:+7.2f} deg/s  tau={tau:+6.3f} Nm")
    gripper = obs.get("gripper.pos", float("nan"))
    print(f"  gripper: pos={gripper:.2f} deg")

    print("[3/4] Micro-motion test (joint_1 +1 deg, then back)...")
    base_pos = obs["joint_1.pos"]
    target = {f"{jn}.pos": obs[f"{jn}.pos"] for jn in joint_names}
    target["gripper.pos"] = obs["gripper.pos"]

    # Move +1 degree
    target["joint_1.pos"] = base_pos + 1.0
    robot.send_action(target)
    time.sleep(0.5)

    obs2 = robot.get_observation()
    delta = obs2["joint_1.pos"] - base_pos
    print(f"  Commanded +1.0 deg, measured delta = {delta:+.2f} deg")

    # Move back
    target["joint_1.pos"] = base_pos
    robot.send_action(target)
    time.sleep(0.5)

    if abs(delta) < 0.3:
        print("  WARNING: small response. Check motor enable / calibration.")
    elif abs(delta - 1.0) < 0.5:
        print("  OK: motor responding correctly")
    else:
        print(f"  WARNING: unexpected delta {delta:.2f}")

    print("[4/4] Parking gripper closed, then disconnecting...")
    _, gripper_close = _openarm_gripper_targets(robot)
    _send_gripper_repeated(robot, gripper_close)
    robot.disconnect()
    print("  OK: pre-flight PASSED\n")


def preflight_piper(port: str, speed_ratio: int = 35):
    from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig

    config = PiperFollowerConfig(
        port=port,
        id="preflight_check",
        speed_ratio=speed_ratio,
        high_follow=False,
        require_calibration=False,
        startup_sleep_s=0.5,
        sync_gripper=True,
    )
    robot = PiperFollower(config)

    print(f"[1/4] Connecting to Piper on {port} (speed_ratio={speed_ratio})...")
    robot.connect()
    print("  OK: connected")

    print("[2/4] Reading joint state...")
    obs = robot.get_observation()
    joint_names = [f"joint_{i}" for i in range(1, 7)]
    for jn in joint_names:
        pos = obs.get(f"{jn}.pos", float("nan"))
        print(f"  {jn}: pos={pos:+7.2f} deg")
    print(f"  gripper: pos={obs.get('gripper.pos', float('nan')):.2f} deg")

    print("[3/4] Micro-motion test (joint_1 +3 deg, then back)...")
    base_pos = obs["joint_1.pos"]
    target = {f"{jn}.pos": obs[f"{jn}.pos"] for jn in joint_names}
    target["gripper.pos"] = obs.get("gripper.pos", 0.0)

    # Piper requires continuous commands — send at 100 Hz for 1.5s
    target["joint_1.pos"] = base_pos + 3.0
    for _ in range(150):
        robot.send_action(target)
        time.sleep(0.01)

    obs2 = robot.get_observation()
    delta = obs2["joint_1.pos"] - base_pos
    print(f"  Commanded +3.0 deg, measured delta = {delta:+.2f} deg")

    # Move back
    target["joint_1.pos"] = base_pos
    for _ in range(150):
        robot.send_action(target)
        time.sleep(0.01)

    if abs(delta) < 0.5:
        print("  WARNING: small response. Check motor enable / connection.")
    elif abs(delta - 3.0) < 1.5:
        print("  OK: motor responding correctly")
    else:
        print(f"  WARNING: unexpected delta {delta:.2f} deg")

    print("[4/4] Disconnecting...")
    robot.disconnect()
    print("  OK: pre-flight PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-flight robot check")
    parser.add_argument("--robot", choices=["openarm", "piper"], required=True)
    parser.add_argument("--port", default="can0", help="CAN interface name")
    parser.add_argument("--side", default="right", help="OpenArm side (left/right)")
    args = parser.parse_args()

    try:
        if args.robot == "openarm":
            preflight_openarm(args.port, args.side)
        else:
            preflight_piper(args.port)
    except Exception as e:
        print(f"\nPRE-FLIGHT FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
