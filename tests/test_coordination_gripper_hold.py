from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from bsreal.experiment.coordination import (
    _escalate_wrist_stabilization_gains,
    _openarm_object_hold_gains,
    _send_gripper_repeated,
)


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


def test_openarm_object_hold_gains_keep_wrist_stiffer_than_gripper() -> None:
    light_kp, light_kd = _openarm_object_hold_gains(is_heavy_object=False)
    heavy_kp, heavy_kd = _openarm_object_hold_gains(is_heavy_object=True)

    assert light_kp["joint_5"] > light_kp["gripper"]
    assert heavy_kp["joint_5"] > light_kp["joint_5"]
    assert heavy_kd["joint_5"] > light_kd["joint_5"]
    assert heavy_kp["gripper"] > light_kp["gripper"]


def test_wrist_stabilization_gain_escalation_preserves_gripper_gain() -> None:
    kp, kd = _escalate_wrist_stabilization_gains(
        {"joint_5": 30.0, "gripper": 8.0},
        {"joint_5": 0.4, "gripper": 0.2},
        attempt_index=2,
    )

    assert kp == {"joint_5": 51.0, "gripper": 8.0}
    assert kd == {"joint_5": 0.68, "gripper": 0.2}
