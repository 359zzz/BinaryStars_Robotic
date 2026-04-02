#!/usr/bin/env python3
"""Interactive joint explorer for Piper dual-arm.

Piper = 6-DOF per arm, position control only.

Keys:
  1-6     Select right arm joint
  l1-l6   Select left arm joint
  +/-     Move selected joint ±5 deg
  ++/--   Move selected joint ±15 deg
  =XX     Set selected joint to XX deg
  m       Mirror right arm → left arm (negate all)
  0       Zero all joints
  r       Read current positions
  s       Save current pose as config
  g       Toggle gripper (open/close)
  q       Quit
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Piper joint explorer")
    parser.add_argument("--right-port", default="can0")
    parser.add_argument("--left-port", default="can1")
    parser.add_argument("--speed-ratio", type=int, default=35)
    args = parser.parse_args()

    from lerobot.robots.bi_piper_follower import (
        BiPiperFollower, BiPiperFollowerConfig,
    )
    from lerobot.robots.piper_follower import PiperFollowerConfig

    config = BiPiperFollowerConfig(
        left_arm_config=PiperFollowerConfig(
            port=args.left_port, id="left", speed_ratio=args.speed_ratio,
        ),
        right_arm_config=PiperFollowerConfig(
            port=args.right_port, id="right", speed_ratio=args.speed_ratio,
        ),
        id="joint_explorer_piper",
    )
    robot = BiPiperFollower(config)
    robot.connect()

    # Current target positions (degrees)
    pos = {}
    for i in range(1, 7):
        pos[f"right_joint_{i}"] = 0.0
        pos[f"left_joint_{i}"] = 0.0

    gripper_open = True  # start open
    selected = "right_joint_1"

    def send_all():
        cmd = {f"{k}.pos": v for k, v in pos.items()}
        # Piper needs all joints in every command
        if gripper_open:
            cmd["right_gripper.pos"] = -50.0
            cmd["left_gripper.pos"] = -50.0
        else:
            cmd["right_gripper.pos"] = 0.0
            cmd["left_gripper.pos"] = 0.0
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
        g_state = "OPEN" if gripper_open else "CLOSED"
        print(f"\n  Gripper: {g_state}")

    def save_pose():
        print("\n  === Saved Pose (copy to trajectory config) ===")
        print("  Right arm:")
        print('  "custom": {')
        for i in range(1, 7):
            val = pos[f"right_joint_{i}"]
            comma = "," if i < 6 else ""
            print(f'      "joint_{i}": {val:.1f}{comma}')
        print("  }")
        print("\n  Left arm positions:")
        for i in range(1, 7):
            print(f"    left_joint_{i} = {pos[f'left_joint_{i}']:.1f}")

    # Send zero position
    send_all()
    time.sleep(1.0)

    print("\n=== Piper Joint Explorer (6-DOF per arm) ===")
    print("Commands:")
    print("  1-6       Select right joint")
    print("  l1-l6     Select left joint")
    print("  + / -     Move ±5 deg")
    print("  ++ / --   Move ±15 deg")
    print("  =XX       Set selected joint to XX deg")
    print("  m         Mirror: copy right arm pose to left arm (negate all)")
    print("  0         Zero all joints")
    print("  r         Read actual positions")
    print("  s         Save current pose")
    print("  g         Toggle gripper open/close")
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
            elif cmd == "g":
                gripper_open = not gripper_open
                send_all()
                state = "OPEN" if gripper_open else "CLOSED"
                print(f"  Gripper → {state}")
            elif cmd == "0":
                for k in pos:
                    pos[k] = 0.0
                send_all()
                print("  All joints zeroed.")
            elif cmd == "m":
                for i in range(1, 7):
                    rval = pos[f"right_joint_{i}"]
                    pos[f"left_joint_{i}"] = -rval
                send_all()
                print("  Left arm mirrored from right arm.")
                read_positions()
            elif cmd == "++":
                pos[selected] += 15.0
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd == "--":
                pos[selected] -= 15.0
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd == "+":
                pos[selected] += 5.0
                send_all()
                print(f"  {selected} → {pos[selected]:.1f}")
            elif cmd == "-":
                pos[selected] -= 5.0
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
                if 1 <= j <= 6:
                    selected = f"left_joint_{j}"
                    print(f"  Selected: {selected}  (pos={pos[selected]:.1f})")
                else:
                    print("  Joint number must be 1-6")
            elif cmd.isdigit() and 1 <= int(cmd) <= 6:
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
        gripper_open = True
        send_all()
        time.sleep(2.0)
        robot.disconnect()
        print("  Done.")


if __name__ == "__main__":
    main()
