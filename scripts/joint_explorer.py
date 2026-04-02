#!/usr/bin/env python3
"""Interactive joint explorer for OpenArm dual-arm.

Controls one joint at a time. Use to figure out joint directions.

Keys:
  1-7     Select right arm joint
  l1-l7   Select left arm joint
  +/-     Move selected joint ±5 deg
  ++/--   Move selected joint ±15 deg
  0       Zero all joints
  r       Read current positions
  s       Save current pose as config
  q       Quit
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive joint explorer")
    parser.add_argument("--right-port", default="can1")
    parser.add_argument("--left-port", default="can2")
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
        id="joint_explorer",
    )
    robot = BiOpenArmFollower(config)
    robot.connect()

    # Current target positions (degrees)
    pos = {}
    for i in range(1, 8):
        pos[f"right_joint_{i}"] = 0.0
        pos[f"left_joint_{i}"] = 0.0

    selected = "right_joint_1"
    step_small = 5.0
    step_big = 15.0

    def send_all():
        cmd = {f"{k}.pos": v for k, v in pos.items()}
        robot.send_action(cmd)

    def read_positions():
        obs = robot.get_observation()
        print("\n  Current positions (deg):")
        print(f"  {'Joint':<18s} {'Target':>8s} {'Actual':>8s}")
        print("  " + "-" * 36)
        for k in sorted(pos.keys()):
            actual = obs.get(f"{k}.pos", float("nan"))
            marker = " <<<" if k == selected else ""
            print(f"  {k:<18s} {pos[k]:>8.1f} {actual:>8.1f}{marker}")

    def save_pose():
        print("\n  === Saved Pose (copy to trajectory.py) ===")
        print('  "custom": {')
        for i in range(1, 8):
            val = pos[f"right_joint_{i}"]
            comma = "," if i < 7 else ""
            print(f'      "joint_{i}": {val:.1f}{comma}')
        print("  }")
        print("\n  Left arm positions (mirrored by SDK):")
        for i in range(1, 8):
            print(f"    left_joint_{i} = {pos[f'left_joint_{i}']:.1f}")

    # Send zero position
    send_all()
    time.sleep(1.0)

    print("\n=== Joint Explorer ===")
    print("Commands:")
    print("  1-7       Select right joint")
    print("  l1-l7     Select left joint")
    print("  + / -     Move ±5 deg")
    print("  ++ / --   Move ±15 deg")
    print("  =XX       Set selected joint to XX deg")
    print("  m         Mirror: copy right arm pose to left arm")
    print("  0         Zero all joints")
    print("  r         Read actual positions")
    print("  s         Save current pose")
    print("  q         Quit")
    print(f"\n  Selected: {selected}  (pos={pos[selected]:.1f})")

    try:
        while True:
            cmd = input(f"\n[{selected} = {pos[selected]:.1f}] > ").strip()

            if cmd == "q":
                break
            elif cmd == "r":
                read_positions()
            elif cmd == "s":
                save_pose()
            elif cmd == "0":
                for k in pos:
                    pos[k] = 0.0
                send_all()
                print("  All joints zeroed.")
            elif cmd == "m":
                # Mirror right → left
                for i in range(1, 8):
                    rval = pos[f"right_joint_{i}"]
                    pos[f"left_joint_{i}"] = -rval
                send_all()
                print("  Left arm mirrored from right arm.")
                read_positions()
            elif cmd == "++":
                pos[selected] += step_big
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd == "--":
                pos[selected] -= step_big
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd == "+":
                pos[selected] += step_small
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd == "-":
                pos[selected] -= step_small
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd.startswith("="):
                try:
                    val = float(cmd[1:])
                    pos[selected] = val
                    send_all()
                    print(f"  {selected} → {pos[selected]:.1f}")
                except ValueError:
                    print("  Invalid number.")
            elif cmd.startswith("l") and len(cmd) >= 2 and cmd[1:].isdigit():
                j = int(cmd[1:])
                if 1 <= j <= 7:
                    selected = f"left_joint_{j}"
                    print(f"  Selected: {selected}  (pos={pos[selected]:.1f})")
                else:
                    print("  Joint number must be 1-7")
            elif cmd.isdigit() and 1 <= int(cmd) <= 7:
                j = int(cmd)
                selected = f"right_joint_{j}"
                print(f"  Selected: {selected}  (pos={pos[selected]:.1f})")
            else:
                print("  Unknown command. Type q to quit, r to read, s to save.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("\n  Zeroing and disconnecting...")
        for k in pos:
            pos[k] = 0.0
        send_all()
        time.sleep(2.0)
        robot.disconnect()
        print("  Done.")


if __name__ == "__main__":
    main()
