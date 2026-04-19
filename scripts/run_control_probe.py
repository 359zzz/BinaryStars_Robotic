#!/usr/bin/env python3
"""Run a controller-sensitive dual-arm probe aligned with a Matrix B probe pair."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsreal.dynamics.effective_mass import make_object_spatial_inertia
from bsreal.experiment.controllers import make_controller
from bsreal.experiment.coordination import (
    GRIPPER_CLOSE_LATCH_STAGES_HEAVY,
    GRIPPER_CLOSE_LATCH_STAGES_LIGHT,
    GRIPPER_HOLD_KD,
    GRIPPER_HOLD_KP,
    GRIPPER_OPEN_LATCH_KD,
    GRIPPER_OPEN_LATCH_KP,
    HEAVY_OBJECT_MASS_KG,
    POST_OBJECT_COOLDOWN_S,
    _close_grippers_with_escalation,
    _current_dual_gripper_cmd,
    _gripper_targets,
    _hold_gripper_target_until_enter,
    _openarm_object_hold_gains,
    _send_gripper_repeated,
    _stabilize_arm_pose_if_needed,
)
from bsreal.experiment.perturbation import _get_observation_with_retry, _send_action_with_retry
from bsreal.experiment.safety import SafetyError, check_position_error, emergency_freeze, slow_move
from bsreal.experiment.trajectory import COORDINATION_CONFIGS, TASK_OBJECTS, get_start_positions_deg
from bsreal.robot_data.openarm_data import make_openarm_dual_arm_ir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPENARM_CONTROLLER_CHOICES = ["decoupled", "j_coupled", "c_coupled", "s_adaptive"]
MAX_PROBE_POSITION_ERROR_DEG = 20.0


def _load_controller_params(raw: str | None) -> dict[str, object]:
    if raw is None:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("controller params JSON must decode to an object")
    return dict(payload)


def _parse_probe_pair(values: list[int], *, n_per_arm: int) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError("probe pair must contain exactly two indices")
    first = int(values[0])
    second = int(values[1])
    if not (0 <= first < n_per_arm and n_per_arm <= second < 2 * n_per_arm):
        raise ValueError(
            "control probe requires a cross-arm pair encoded as [right_idx, left_idx+n_per_arm]"
        )
    return first, second


def _joint_name_lists(n_per_arm: int) -> tuple[list[str], list[str], list[str]]:
    right = [f"right_joint_{index}" for index in range(1, n_per_arm + 1)]
    left = [f"left_joint_{index}" for index in range(1, n_per_arm + 1)]
    return right, left, right + left


def _make_robot(args: argparse.Namespace):
    if args.dry_run:
        return None

    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
    from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

    config = BiOpenArmFollowerConfig(
        left_arm_config=OpenArmFollowerConfigBase(port=args.left_port, side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port=args.right_port, side="right"),
        id="control_probe_exp",
    )
    return BiOpenArmFollower(config)


def _base_target_deg(config_name: str, all_joint_names: list[str]) -> np.ndarray:
    start = get_start_positions_deg(config_name)
    return np.asarray([float(start[joint_name]) for joint_name in all_joint_names], dtype=np.float64)


def _prepare_robot_start(
    robot,
    *,
    base_target_deg: np.ndarray,
    all_joint_names: list[str],
) -> None:
    cmd = {
        f"{joint_name}.pos": float(base_target_deg[index])
        for index, joint_name in enumerate(all_joint_names)
    }
    cmd.update(_current_dual_gripper_cmd(robot))
    slow_move(
        robot,
        cmd,
        duration_s=3.0,
        custom_kp=GRIPPER_HOLD_KP,
        custom_kd=GRIPPER_HOLD_KD,
    )
    time.sleep(1.0)


def _prepare_object_if_needed(
    robot,
    *,
    object_mass_kg: float,
    arm_hold_cmd: dict[str, float],
) -> tuple[bool, dict[str, float], tuple[float, float]]:
    has_object = object_mass_kg > 0.0
    gripper_open, gripper_close = _gripper_targets(robot, "openarm")
    active_gripper_cmd = _current_dual_gripper_cmd(robot)
    if not has_object:
        return False, active_gripper_cmd, (gripper_open, gripper_close)

    is_heavy_object = object_mass_kg >= HEAVY_OBJECT_MASS_KG
    hold_kp, hold_kd = _openarm_object_hold_gains(is_heavy_object=is_heavy_object)
    close_stages = (
        GRIPPER_CLOSE_LATCH_STAGES_HEAVY if is_heavy_object else GRIPPER_CLOSE_LATCH_STAGES_LIGHT
    )

    _send_gripper_repeated(
        robot,
        gripper_open,
        duration_s=0.5,
        custom_kp=GRIPPER_OPEN_LATCH_KP,
        custom_kd=GRIPPER_OPEN_LATCH_KD,
        arm_hold_cmd=arm_hold_cmd,
    )
    _hold_gripper_target_until_enter(
        robot,
        gripper_open,
        (
            f"\n>>> Grippers are being held open. Place the bar "
            f"({object_mass_kg} kg) between both grippers, then press ENTER..."
        ),
        custom_kp=hold_kp,
        custom_kd=hold_kd,
        arm_hold_cmd=arm_hold_cmd,
    )
    print("  Grippers closing...")
    active_gripper_cmd = _close_grippers_with_escalation(
        robot,
        open_target=gripper_open,
        close_target=gripper_close,
        close_stages=close_stages,
        open_latch_kp=GRIPPER_OPEN_LATCH_KP,
        open_latch_kd=GRIPPER_OPEN_LATCH_KD,
        hold_kp=hold_kp,
        hold_kd=hold_kd,
        arm_hold_cmd=arm_hold_cmd,
    )
    settled_cmd = dict(arm_hold_cmd)
    settled_cmd.update(active_gripper_cmd)
    slow_move(
        robot,
        settled_cmd,
        duration_s=1.0,
        custom_kp=hold_kp,
        custom_kd=hold_kd,
    )
    time.sleep(0.3)
    _stabilize_arm_pose_if_needed(
        robot,
        arm_hold_cmd=arm_hold_cmd,
        active_gripper_cmd=active_gripper_cmd,
        custom_kp=hold_kp,
        custom_kd=hold_kd,
    )
    return True, active_gripper_cmd, (gripper_open, gripper_close)


def _release_object_if_needed(
    robot,
    *,
    has_object: bool,
    object_mass_kg: float,
    active_gripper_cmd: dict[str, float],
    gripper_targets: tuple[float, float],
    arm_hold_cmd: dict[str, float],
) -> dict[str, float]:
    if not has_object:
        return active_gripper_cmd

    gripper_open, gripper_close = gripper_targets
    is_heavy_object = object_mass_kg >= HEAVY_OBJECT_MASS_KG
    hold_kp, hold_kd = _openarm_object_hold_gains(is_heavy_object=is_heavy_object)

    _hold_gripper_target_until_enter(
        robot,
        gripper_close,
        "\n>>> Support the bar, then press ENTER to release the grippers...",
        custom_kp=hold_kp,
        custom_kd=hold_kd,
        arm_hold_cmd=arm_hold_cmd,
    )
    _send_gripper_repeated(
        robot,
        gripper_open,
        duration_s=0.8,
        custom_kp=GRIPPER_OPEN_LATCH_KP,
        custom_kd=GRIPPER_OPEN_LATCH_KD,
        arm_hold_cmd=arm_hold_cmd,
    )
    time.sleep(0.3)
    _hold_gripper_target_until_enter(
        robot,
        gripper_open,
        "\n>>> Remove the bar completely, then press ENTER to park the empty grippers closed...",
        custom_kp=hold_kp,
        custom_kd=hold_kd,
        arm_hold_cmd=arm_hold_cmd,
    )
    active_gripper_cmd = _send_gripper_repeated(
        robot,
        gripper_close,
        duration_s=0.8,
        custom_kp=None,
        custom_kd=None,
        arm_hold_cmd=arm_hold_cmd,
    )
    logger.info("Cooling grippers for %.1fs before finishing.", POST_OBJECT_COOLDOWN_S)
    time.sleep(POST_OBJECT_COOLDOWN_S)
    return active_gripper_cmd


def _direction_result(
    *,
    direction_id: str,
    perturbed_full_idx: int,
    opposite_full_idx: int,
    timestamps_s: list[float],
    target_series_deg: list[np.ndarray],
    actual_series_deg: list[np.ndarray],
    input_offsets_deg: list[float],
) -> dict[str, object]:
    target = np.asarray(target_series_deg, dtype=np.float64)
    actual = np.asarray(actual_series_deg, dtype=np.float64)
    offsets = np.asarray(input_offsets_deg, dtype=np.float64)
    perturbed_error = actual[:, perturbed_full_idx] - target[:, perturbed_full_idx]
    opposite_error = actual[:, opposite_full_idx] - target[:, opposite_full_idx]
    full_error = actual - target

    corr = 0.0
    if len(offsets) > 1 and np.std(offsets) > 1e-9 and np.std(opposite_error) > 1e-9:
        corr = float(np.corrcoef(offsets, opposite_error)[0, 1])
    amplitude = max(float(np.max(np.abs(offsets))), 1.0e-9)
    gain = float(np.max(np.abs(opposite_error)) / amplitude)

    return {
        "direction_id": direction_id,
        "perturbed_full_index": int(perturbed_full_idx),
        "opposite_full_index": int(opposite_full_idx),
        "n_samples": len(timestamps_s),
        "perturbed_joint_tracking_rmse_deg": float(np.sqrt(np.mean(perturbed_error**2))),
        "opposite_probe_hold_rmse_deg": float(np.sqrt(np.mean(opposite_error**2))),
        "full_body_rmse_deg": float(np.sqrt(np.mean(full_error**2))),
        "opposite_probe_peak_abs_error_deg": float(np.max(np.abs(opposite_error))),
        "coupling_response_corr": corr,
        "coupling_response_gain": gain,
        "timestamps_s": [float(value) for value in timestamps_s],
        "perturbed_target_deg": [float(value) for value in target[:, perturbed_full_idx]],
        "perturbed_actual_deg": [float(value) for value in actual[:, perturbed_full_idx]],
        "opposite_target_deg": [float(value) for value in target[:, opposite_full_idx]],
        "opposite_actual_deg": [float(value) for value in actual[:, opposite_full_idx]],
    }


def _run_directional_probe(
    robot,
    controller,
    *,
    direction_id: str,
    perturbed_full_idx: int,
    opposite_full_idx: int,
    all_joint_names: list[str],
    base_target_deg: np.ndarray,
    active_gripper_cmd: dict[str, float],
    has_object: bool,
    object_mass_kg: float,
    amplitude_deg: float,
    frequency_hz: float,
    duration_s: float,
    dt: float,
    ramp_s: float,
) -> dict[str, object]:
    timestamps_s: list[float] = []
    target_series_deg: list[np.ndarray] = []
    actual_series_deg: list[np.ndarray] = []
    input_offsets_deg: list[float] = []

    is_heavy_object = object_mass_kg >= HEAVY_OBJECT_MASS_KG
    hold_kp, hold_kd = _openarm_object_hold_gains(is_heavy_object=is_heavy_object)
    custom_kp = hold_kp if has_object else None
    custom_kd = hold_kd if has_object else None
    qdot_target = np.zeros_like(base_target_deg)
    omega = 2.0 * math.pi * frequency_hz
    t0 = time.monotonic()
    safety_errors = 0

    while True:
        t = time.monotonic() - t0
        if t >= duration_s:
            break

        amp = amplitude_deg * min(t / max(ramp_s, 1.0e-6), 1.0)
        offset = amp * math.sin(omega * t)
        q_target_deg = base_target_deg.copy()
        q_target_deg[perturbed_full_idx] += offset
        q_target_rad = np.deg2rad(q_target_deg)

        obs = _get_observation_with_retry(robot, label=f"{direction_id}:get_observation")
        q_current_deg = np.asarray(
            [obs.get(f"{joint_name}.pos", 0.0) for joint_name in all_joint_names],
            dtype=np.float64,
        )
        q_current_rad = np.deg2rad(q_current_deg)
        qdot_current_deg_s = np.asarray(
            [obs.get(f"{joint_name}.vel", 0.0) for joint_name in all_joint_names],
            dtype=np.float64,
        )
        qdot_current_rad_s = np.deg2rad(qdot_current_deg_s)
        tau_ff = controller.compute_action(
            t,
            q_current_rad,
            qdot_current_rad_s,
            q_target_rad,
            qdot_target,
        )

        cmd = {}
        for index, joint_name in enumerate(all_joint_names):
            cmd[f"{joint_name}.pos"] = float(q_target_deg[index])
            cmd[f"{joint_name}.tau_ff"] = float(tau_ff[index])
        if has_object:
            cmd.update(active_gripper_cmd)
        _send_action_with_retry(
            robot,
            cmd,
            custom_kp=custom_kp,
            custom_kd=custom_kd,
            label=f"{direction_id}:send_action",
        )

        timestamps_s.append(float(t))
        target_series_deg.append(q_target_deg)
        actual_series_deg.append(q_current_deg)
        input_offsets_deg.append(float(offset))

        try:
            check_position_error(
                {joint_name: float(q_current_deg[index]) for index, joint_name in enumerate(all_joint_names)},
                {joint_name: float(q_target_deg[index]) for index, joint_name in enumerate(all_joint_names)},
                MAX_PROBE_POSITION_ERROR_DEG,
            )
            safety_errors = 0
        except SafetyError as exc:
            safety_errors += 1
            if safety_errors >= 5:
                logger.error("Safety triggered during control probe: %s", exc)
                emergency_freeze(robot)
                raise

        sleep_time = dt - (time.monotonic() - t0 - t)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    base_cmd = {
        f"{joint_name}.pos": float(base_target_deg[index])
        for index, joint_name in enumerate(all_joint_names)
    }
    if has_object:
        base_cmd.update(active_gripper_cmd)
    _send_action_with_retry(
        robot,
        base_cmd,
        custom_kp=custom_kp,
        custom_kd=custom_kd,
        label=f"{direction_id}:return_to_base",
    )
    time.sleep(0.5)
    return _direction_result(
        direction_id=direction_id,
        perturbed_full_idx=perturbed_full_idx,
        opposite_full_idx=opposite_full_idx,
        timestamps_s=timestamps_s,
        target_series_deg=target_series_deg,
        actual_series_deg=actual_series_deg,
        input_offsets_deg=input_offsets_deg,
    )


def _simulate_directional_probe(
    *,
    direction_id: str,
    perturbed_full_idx: int,
    opposite_full_idx: int,
    base_target_deg: np.ndarray,
    amplitude_deg: float,
    frequency_hz: float,
    duration_s: float,
    dt: float,
    ramp_s: float,
) -> dict[str, object]:
    timestamps_s: list[float] = []
    target_series_deg: list[np.ndarray] = []
    actual_series_deg: list[np.ndarray] = []
    input_offsets_deg: list[float] = []
    omega = 2.0 * math.pi * frequency_hz
    for t in np.arange(0.0, duration_s, dt):
        amp = amplitude_deg * min(float(t) / max(ramp_s, 1.0e-6), 1.0)
        offset = amp * math.sin(omega * float(t))
        q_target_deg = base_target_deg.copy()
        q_target_deg[perturbed_full_idx] += offset
        q_actual_deg = q_target_deg.copy()
        q_actual_deg[perturbed_full_idx] += 0.25 * math.sin(omega * float(t) + 0.05)
        q_actual_deg[opposite_full_idx] += 0.10 * math.sin(omega * float(t) + 0.2)
        timestamps_s.append(float(t))
        target_series_deg.append(q_target_deg)
        actual_series_deg.append(q_actual_deg)
        input_offsets_deg.append(float(offset))
    return _direction_result(
        direction_id=direction_id,
        perturbed_full_idx=perturbed_full_idx,
        opposite_full_idx=opposite_full_idx,
        timestamps_s=timestamps_s,
        target_series_deg=target_series_deg,
        actual_series_deg=actual_series_deg,
        input_offsets_deg=input_offsets_deg,
    )


def _aggregate_direction_metrics(directions: list[dict[str, object]]) -> dict[str, object]:
    if not directions:
        return {
            "direction_count": 0,
            "mean_full_body_rmse_deg": 0.0,
            "mean_opposite_probe_hold_rmse_deg": 0.0,
            "mean_perturbed_joint_tracking_rmse_deg": 0.0,
            "max_opposite_probe_peak_abs_error_deg": 0.0,
        }

    return {
        "direction_count": len(directions),
        "mean_full_body_rmse_deg": float(
            np.mean([float(item["full_body_rmse_deg"]) for item in directions])
        ),
        "mean_opposite_probe_hold_rmse_deg": float(
            np.mean([float(item["opposite_probe_hold_rmse_deg"]) for item in directions])
        ),
        "mean_perturbed_joint_tracking_rmse_deg": float(
            np.mean([float(item["perturbed_joint_tracking_rmse_deg"]) for item in directions])
        ),
        "max_opposite_probe_peak_abs_error_deg": float(
            max(float(item["opposite_probe_peak_abs_error_deg"]) for item in directions)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controller-sensitive dual-arm control probe")
    parser.add_argument("--robot", choices=["openarm"], default="openarm")
    parser.add_argument("--left-port", default="can1")
    parser.add_argument("--right-port", default="can0")
    parser.add_argument("--task", choices=list(TASK_OBJECTS.keys()), default="bar_loaded")
    parser.add_argument("--config", choices=list(COORDINATION_CONFIGS.keys()), default="bar_b")
    parser.add_argument("--controller", choices=OPENARM_CONTROLLER_CHOICES, required=True)
    parser.add_argument("--controller-params-json", default=None)
    parser.add_argument("--probe-pair", nargs=2, type=int, required=True)
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--ramp-s", type=float, default=2.0)
    parser.add_argument("--output-dir", default="results/control_probe")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    n_per_arm = 7
    probe_right_idx, probe_left_full_idx = _parse_probe_pair(args.probe_pair, n_per_arm=n_per_arm)
    probe_left_idx = probe_left_full_idx - n_per_arm
    controller_params = _load_controller_params(args.controller_params_json)
    ir = make_openarm_dual_arm_ir()
    _, _, all_joint_names = _joint_name_lists(n_per_arm)
    base_target_deg = _base_target_deg(args.config, all_joint_names)
    arm_hold_cmd = {
        f"{joint_name}.pos": float(base_target_deg[index])
        for index, joint_name in enumerate(all_joint_names)
    }

    directions: list[dict[str, object]]
    robot = _make_robot(args)
    if args.dry_run:
        directions = [
            _simulate_directional_probe(
                direction_id="right_to_left",
                perturbed_full_idx=probe_right_idx,
                opposite_full_idx=n_per_arm + probe_left_idx,
                base_target_deg=base_target_deg,
                amplitude_deg=args.amplitude,
                frequency_hz=args.frequency,
                duration_s=args.duration,
                dt=args.dt,
                ramp_s=args.ramp_s,
            ),
            _simulate_directional_probe(
                direction_id="left_to_right",
                perturbed_full_idx=n_per_arm + probe_left_idx,
                opposite_full_idx=probe_right_idx,
                base_target_deg=base_target_deg,
                amplitude_deg=args.amplitude,
                frequency_hz=args.frequency,
                duration_s=args.duration,
                dt=args.dt,
                ramp_s=args.ramp_s,
            ),
        ]
    else:
        obj = TASK_OBJECTS[args.task]
        M_obj = None
        if obj["mass"] > 0.0:
            M_obj = make_object_spatial_inertia(obj["mass"], obj["geometry"], obj["dims"])
        is_heavy_object = float(obj["mass"]) >= HEAVY_OBJECT_MASS_KG
        object_hold_kp, object_hold_kd = _openarm_object_hold_gains(
            is_heavy_object=is_heavy_object
        )

        robot.connect()
        active_gripper_cmd: dict[str, float] = {}
        gripper_targets = (-65.0, 0.0)
        has_object = False
        try:
            _prepare_robot_start(
                robot,
                base_target_deg=base_target_deg,
                all_joint_names=all_joint_names,
            )
            has_object, active_gripper_cmd, gripper_targets = _prepare_object_if_needed(
                robot,
                object_mass_kg=float(obj["mass"]),
                arm_hold_cmd=arm_hold_cmd,
            )
            if has_object:
                _stabilize_arm_pose_if_needed(
                    robot,
                    arm_hold_cmd=arm_hold_cmd,
                    active_gripper_cmd=active_gripper_cmd,
                    custom_kp=object_hold_kp,
                    custom_kd=object_hold_kd,
                )
            directions = [
                _run_directional_probe(
                    robot,
                    make_controller(
                        args.controller,
                        ir,
                        n_per_arm,
                        "openarm",
                        M_obj=M_obj,
                        **controller_params,
                    ),
                    direction_id="right_to_left",
                    perturbed_full_idx=probe_right_idx,
                    opposite_full_idx=n_per_arm + probe_left_idx,
                    all_joint_names=all_joint_names,
                    base_target_deg=base_target_deg,
                    active_gripper_cmd=active_gripper_cmd,
                    has_object=has_object,
                    object_mass_kg=float(obj["mass"]),
                    amplitude_deg=args.amplitude,
                    frequency_hz=args.frequency,
                    duration_s=args.duration,
                    dt=args.dt,
                    ramp_s=args.ramp_s,
                ),
                _run_directional_probe(
                    robot,
                    make_controller(
                        args.controller,
                        ir,
                        n_per_arm,
                        "openarm",
                        M_obj=M_obj,
                        **controller_params,
                    ),
                    direction_id="left_to_right",
                    perturbed_full_idx=n_per_arm + probe_left_idx,
                    opposite_full_idx=probe_right_idx,
                    all_joint_names=all_joint_names,
                    base_target_deg=base_target_deg,
                    active_gripper_cmd=active_gripper_cmd,
                    has_object=has_object,
                    object_mass_kg=float(obj["mass"]),
                    amplitude_deg=args.amplitude,
                    frequency_hz=args.frequency,
                    duration_s=args.duration,
                    dt=args.dt,
                    ramp_s=args.ramp_s,
                ),
            ]
        finally:
            try:
                active_gripper_cmd = _release_object_if_needed(
                    robot,
                    has_object=has_object,
                    object_mass_kg=float(TASK_OBJECTS[args.task]["mass"]),
                    active_gripper_cmd=active_gripper_cmd,
                    gripper_targets=gripper_targets,
                    arm_hold_cmd=arm_hold_cmd,
                )
                final_cmd = {
                    f"{joint_name}.pos": float(base_target_deg[index])
                    for index, joint_name in enumerate(all_joint_names)
                }
                if active_gripper_cmd:
                    final_cmd.update(active_gripper_cmd)
                slow_move(
                    robot,
                    final_cmd,
                    duration_s=3.0,
                    custom_kp=GRIPPER_HOLD_KP,
                    custom_kd=GRIPPER_HOLD_KD,
                )
            except Exception:
                logger.exception("Failed to park robot cleanly after control probe")
            try:
                robot.disconnect()
            except Exception:
                logger.exception("Failed to disconnect robot cleanly after control probe")

    payload = {
        "schema": "matrix_d_control_probe_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "robot": args.robot,
        "task": args.task,
        "config": args.config,
        "controller": args.controller,
        "controller_params": controller_params,
        "probe_pair": [probe_right_idx, probe_left_full_idx],
        "probe_right_joint_idx": probe_right_idx,
        "probe_left_joint_idx": probe_left_idx,
        "amplitude_deg": float(args.amplitude),
        "frequency_hz": float(args.frequency),
        "duration_s": float(args.duration),
        "dt": float(args.dt),
        "dry_run": bool(args.dry_run),
        "directions": directions,
        "aggregate": _aggregate_direction_metrics(directions),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"control_probe_pair_{probe_right_idx}_{probe_left_full_idx}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Saved: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
