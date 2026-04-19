from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from bsreal.experiment.coordination import _send_gripper_repeated


class _DummyRobot:
    def __init__(self) -> None:
        self.commands: list[tuple[dict[str, float], dict[str, float] | None, dict[str, float] | None]] = []

    def send_action(self, cmd, custom_kp=None, custom_kd=None) -> None:
        self.commands.append((dict(cmd), custom_kp, custom_kd))


def test_send_gripper_repeated_keeps_arm_hold_pose() -> None:
    robot = _DummyRobot()
    arm_hold_cmd = {
        "right_joint_5.pos": 75.0,
        "left_joint_5.pos": -75.0,
    }

    gripper_cmd = _send_gripper_repeated(
        robot,
        -55.0,
        duration_s=0.02,
        dt=0.01,
        arm_hold_cmd=arm_hold_cmd,
    )

    assert gripper_cmd == {
        "right_gripper.pos": -55.0,
        "left_gripper.pos": -55.0,
    }
    assert robot.commands
    for cmd, _, _ in robot.commands:
        assert cmd["right_joint_5.pos"] == 75.0
        assert cmd["left_joint_5.pos"] == -75.0
        assert cmd["right_gripper.pos"] == -55.0
        assert cmd["left_gripper.pos"] == -55.0
