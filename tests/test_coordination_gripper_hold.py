from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from bsreal.experiment.coordination import (
    _adapt_contact_settled_passive_joints,
    _apply_arm_hold_overrides_to_target_matrix,
    _escalate_wrist_stabilization_gains,
    _openarm_object_hold_gains,
    _send_gripper_repeated,
)


class _DummyRobot:
    def __init__(self, observation: dict[str, float] | None = None) -> None:
        self.commands: list[tuple[dict[str, float], dict[str, float] | None, dict[str, float] | None]] = []
        self.observation = observation or {}

    def send_action(self, cmd, custom_kp=None, custom_kd=None) -> None:
        self.commands.append((dict(cmd), custom_kp, custom_kd))

    def get_observation(self) -> dict[str, float]:
        return dict(self.observation)


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


def test_contact_settled_passive_joint_adapts_only_large_wrist_drift() -> None:
    robot = _DummyRobot(
        {
            "right_joint_5.pos": 73.0,
            "left_joint_5.pos": -6.4,
            "left_joint_6.pos": 0.0,
        }
    )
    arm_hold_cmd = {
        "right_joint_5.pos": 75.0,
        "left_joint_5.pos": -75.0,
        "left_joint_6.pos": -25.0,
    }

    adapted = _adapt_contact_settled_passive_joints(robot, arm_hold_cmd)

    assert adapted == {"left_joint_5.pos": -6.4}
    assert arm_hold_cmd["right_joint_5.pos"] == 75.0
    assert arm_hold_cmd["left_joint_5.pos"] == -6.4
    assert arm_hold_cmd["left_joint_6.pos"] == -25.0


def test_apply_arm_hold_overrides_to_target_matrix() -> None:
    target = np.zeros((3, 4))
    all_joint_names = ["right_joint_5", "right_joint_6", "left_joint_5", "left_joint_6"]
    arm_hold_cmd = {"left_joint_5.pos": -6.4, "right_joint_5.pos": 75.0}

    _apply_arm_hold_overrides_to_target_matrix(
        target,
        all_joint_names,
        arm_hold_cmd,
        {"left_joint_5.pos"},
    )

    assert np.allclose(target[:, 0], 0.0)
    assert np.allclose(target[:, 2], -6.4)
