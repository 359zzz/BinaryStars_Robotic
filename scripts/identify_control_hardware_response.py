#!/usr/bin/env python3
"""Identify closed-loop hardware response matrices for Matrix B control rounds.

The generated JSON is intentionally shaped so BinaryStars Matrix B can consume
it directly through ``--hardware-response-input``:

{
  "bar_only:bar_b": {
    "decoupled_ref": {"response_matrix": [[... 14x14 ...]], ...},
    ...
  },
  "bar_loaded:bar_b": {...}
}

Each matrix column is identified by perturbing one target joint with a small
sinusoid and fitting the observed closed-loop position response of all joints:

    R[:, j] ~= d q_actual / d q_target_j

This is a real-robot closed-loop response model, not a QPU calibration object.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsreal.dynamics.effective_mass import make_object_spatial_inertia
from bsreal.experiment.controllers import make_controller
from bsreal.experiment.coordination import (
    GRIPPER_HOLD_KD,
    GRIPPER_HOLD_KP,
    HEAVY_OBJECT_MASS_KG,
    POST_OBJECT_COOLDOWN_S,
    _adapt_contact_settled_passive_joints,
    _openarm_object_hold_gains,
    _stabilize_arm_pose_if_needed,
)
from bsreal.experiment.perturbation import _get_observation_with_retry, _send_action_with_retry
from bsreal.experiment.safety import SafetyError, check_position_error, emergency_freeze, slow_move
from bsreal.experiment.trajectory import COORDINATION_CONFIGS, TASK_OBJECTS, get_start_positions_deg
from bsreal.robot_data.openarm_data import make_openarm_dual_arm_ir
from scripts.run_control_probe import (
    _joint_name_lists,
    _make_robot,
    _prepare_object_if_needed,
    _prepare_robot_start,
    _release_object_if_needed,
    _sync_base_target_from_arm_hold_cmd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = "matrix_b_hardware_response_models_v1"
RESPONSE_SOURCE = "hardware_identified_v1"
RESPONSE_AXIS = "closed_loop_position_response_deg_per_deg_target_v1"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "matrix_b_hardware_response"
N_PER_ARM = 7
N_TOTAL = 2 * N_PER_ARM
MAX_IDENTIFICATION_POSITION_ERROR_DEG = 25.0

DEFAULT_CANDIDATES: dict[str, tuple[str, dict[str, object]]] = {
    "decoupled_ref": ("decoupled", {}),
    "j_coupled_eng": (
        "j_coupled",
        {"kp_comp": 2.0, "kd_comp": 0.1, "alpha_pos": 0.3},
    ),
    "c_coupled_cross": (
        "c_coupled",
        {"kp_comp": 2.0, "kd_comp": 0.1, "alpha_pos": 0.3},
    ),
    "s_adaptive_entropy": (
        "s_adaptive",
        {
            "kp_comp": 2.0,
            "kd_comp": 0.1,
            "alpha_pos": 0.3,
            "s_threshold": 1.75,
            "transition_width": 0.5,
            "recompute_every": 10,
        },
    ),
}


@dataclass(frozen=True, slots=True)
class ContextSpec:
    task_name: str
    config_name: str

    @property
    def context_id(self) -> str:
        return f"{self.task_name}:{self.config_name}"


def _parse_context(raw: str) -> ContextSpec:
    if ":" not in raw:
        raise ValueError(f"invalid context {raw!r}; expected task:config")
    task_name, config_name = raw.split(":", maxsplit=1)
    if task_name not in TASK_OBJECTS:
        raise ValueError(f"unknown task {task_name!r}; expected {sorted(TASK_OBJECTS)}")
    if config_name not in COORDINATION_CONFIGS:
        raise ValueError(f"unknown config {config_name!r}; expected {sorted(COORDINATION_CONFIGS)}")
    return ContextSpec(task_name=task_name, config_name=config_name)


def _parse_controller_params(raw: str | None) -> dict[str, dict[str, object]]:
    if raw is None:
        return {}
    payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--controller-params-input must decode to an object")
    result: dict[str, dict[str, object]] = {}
    for candidate_id, params in payload.items():
        if isinstance(params, dict):
            result[str(candidate_id)] = dict(params)
    return result


def _base_target_deg(config_name: str, all_joint_names: list[str]) -> np.ndarray:
    start = get_start_positions_deg(config_name)
    return np.asarray([float(start[joint_name]) for joint_name in all_joint_names], dtype=np.float64)


def _candidate_specs(
    candidate_ids: list[str],
    *,
    controller_params_override: dict[str, dict[str, object]],
) -> dict[str, tuple[str, dict[str, object]]]:
    specs: dict[str, tuple[str, dict[str, object]]] = {}
    for candidate_id in candidate_ids:
        if candidate_id not in DEFAULT_CANDIDATES:
            raise ValueError(f"unknown candidate_id {candidate_id!r}; expected {sorted(DEFAULT_CANDIDATES)}")
        controller_name, default_params = DEFAULT_CANDIDATES[candidate_id]
        params = {**default_params, **controller_params_override.get(candidate_id, {})}
        specs[candidate_id] = (controller_name, params)
    return specs


def _make_object_inertia(task_name: str):
    obj = TASK_OBJECTS[task_name]
    if float(obj["mass"]) <= 0.0:
        return None
    return make_object_spatial_inertia(float(obj["mass"]), str(obj["geometry"]), tuple(obj["dims"]))


def _fit_response_column(
    *,
    input_offsets_deg: list[float],
    actual_series_deg: list[list[float]],
    base_target_deg: np.ndarray,
    settle_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    offsets = np.asarray(input_offsets_deg, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_series_deg, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] != base_target_deg.size:
        raise ValueError("actual_series_deg must be a T x n_total matrix")
    if actual.shape[0] != offsets.size:
        raise ValueError("input_offsets_deg length must match actual_series_deg")
    start = min(max(int(math.floor(offsets.size * float(settle_fraction))), 0), offsets.size - 1)
    x = offsets[start:]
    y = actual[start:, :] - np.asarray(base_target_deg, dtype=np.float64).reshape(1, -1)
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1.0e-9:
        raise ValueError("insufficient input variation to fit response column")
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    column = (x_centered.reshape(1, -1) @ y_centered / denom).reshape(-1)
    pred = x_centered.reshape(-1, 1) * column.reshape(1, -1)
    residual = y_centered - pred
    metrics = {
        "fit_start_index": float(start),
        "fit_sample_count": float(x.size),
        "input_std_deg": float(np.std(x)),
        "mean_abs_residual_deg": float(np.mean(np.abs(residual))),
        "max_abs_residual_deg": float(np.max(np.abs(residual))),
    }
    return column.astype(np.float64), metrics


def _response_matrix_from_column_trials(
    column_trials: list[dict[str, object]],
    *,
    n_total: int,
    base_target_deg: np.ndarray,
    settle_fraction: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    response = np.zeros((n_total, n_total), dtype=np.float64)
    fit_metrics: list[dict[str, object]] = []
    for trial in column_trials:
        column_index = int(trial["perturbed_full_index"])
        column, metrics = _fit_response_column(
            input_offsets_deg=[float(value) for value in trial["input_offsets_deg"]],
            actual_series_deg=[
                [float(item) for item in row]
                for row in trial["actual_series_deg"]  # type: ignore[index]
            ],
            base_target_deg=base_target_deg,
            settle_fraction=settle_fraction,
        )
        response[:, column_index] = column
        fit_metrics.append({"column_index": column_index, **metrics})
    return response, fit_metrics


def _simulate_response_template(candidate_id: str, n_total: int) -> np.ndarray:
    response = np.eye(n_total, dtype=np.float64)
    if candidate_id == "j_coupled_eng":
        response[2, 10] = 0.04
        response[10, 2] = 0.035
    elif candidate_id == "c_coupled_cross":
        response[2, 10] = 0.14
        response[10, 2] = 0.13
        response[6, 13] = 0.18
        response[13, 6] = 0.16
    elif candidate_id == "s_adaptive_entropy":
        response[2, 10] = 0.08
        response[10, 2] = 0.075
        response[6, 13] = 0.11
        response[13, 6] = 0.10
    return response


def _simulate_column_trial(
    *,
    candidate_id: str,
    column_index: int,
    base_target_deg: np.ndarray,
    amplitude_deg: float,
    frequency_hz: float,
    duration_s: float,
    dt: float,
    ramp_s: float,
) -> dict[str, object]:
    response = _simulate_response_template(candidate_id, base_target_deg.size)
    omega = 2.0 * math.pi * float(frequency_hz)
    timestamps: list[float] = []
    input_offsets: list[float] = []
    actual_series: list[list[float]] = []
    target_series: list[list[float]] = []
    for t in np.arange(0.0, float(duration_s), float(dt)):
        amp = float(amplitude_deg) * min(float(t) / max(float(ramp_s), 1.0e-6), 1.0)
        offset = amp * math.sin(omega * float(t))
        target = base_target_deg.copy()
        target[column_index] += offset
        actual = base_target_deg + response[:, column_index] * offset
        timestamps.append(float(t))
        input_offsets.append(float(offset))
        target_series.append([float(value) for value in target])
        actual_series.append([float(value) for value in actual])
    return {
        "perturbed_full_index": int(column_index),
        "timestamps_s": timestamps,
        "input_offsets_deg": input_offsets,
        "target_series_deg": target_series,
        "actual_series_deg": actual_series,
        "dry_run": True,
    }


def _run_column_probe(
    robot,
    controller,
    *,
    column_index: int,
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
    timestamps: list[float] = []
    input_offsets: list[float] = []
    actual_series: list[list[float]] = []
    target_series: list[list[float]] = []
    is_heavy_object = object_mass_kg >= HEAVY_OBJECT_MASS_KG
    hold_kp, hold_kd = _openarm_object_hold_gains(is_heavy_object=is_heavy_object)
    custom_kp = hold_kp if has_object else None
    custom_kd = hold_kd if has_object else None
    qdot_target = np.zeros_like(base_target_deg)
    omega = 2.0 * math.pi * float(frequency_hz)
    t0 = time.monotonic()
    safety_errors = 0

    while True:
        t = time.monotonic() - t0
        if t >= duration_s:
            break

        amp = float(amplitude_deg) * min(t / max(float(ramp_s), 1.0e-6), 1.0)
        offset = amp * math.sin(omega * t)
        q_target_deg = base_target_deg.copy()
        q_target_deg[column_index] += offset
        q_target_rad = np.deg2rad(q_target_deg)

        obs = _get_observation_with_retry(robot, label=f"hardware_response_col_{column_index}:get_observation")
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

        cmd: dict[str, float] = {}
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
            label=f"hardware_response_col_{column_index}:send_action",
        )

        timestamps.append(float(t))
        input_offsets.append(float(offset))
        target_series.append([float(value) for value in q_target_deg])
        actual_series.append([float(value) for value in q_current_deg])

        try:
            check_position_error(
                {joint_name: float(q_current_deg[index]) for index, joint_name in enumerate(all_joint_names)},
                {joint_name: float(q_target_deg[index]) for index, joint_name in enumerate(all_joint_names)},
                MAX_IDENTIFICATION_POSITION_ERROR_DEG,
            )
            safety_errors = 0
        except SafetyError:
            safety_errors += 1
            if safety_errors >= 5:
                logger.exception("Safety triggered during hardware response identification")
                emergency_freeze(robot)
                raise

        sleep_time = float(dt) - (time.monotonic() - t0 - t)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    base_cmd = {f"{joint_name}.pos": float(base_target_deg[index]) for index, joint_name in enumerate(all_joint_names)}
    if has_object:
        base_cmd.update(active_gripper_cmd)
    _send_action_with_retry(
        robot,
        base_cmd,
        custom_kp=custom_kp,
        custom_kd=custom_kd,
        label=f"hardware_response_col_{column_index}:return_to_base",
    )
    time.sleep(0.4)
    return {
        "perturbed_full_index": int(column_index),
        "timestamps_s": timestamps,
        "input_offsets_deg": input_offsets,
        "target_series_deg": target_series,
        "actual_series_deg": actual_series,
        "dry_run": False,
    }


def _matrix_payload(
    *,
    response_matrix: np.ndarray,
    candidate_id: str,
    controller_name: str,
    controller_params: dict[str, object],
    context: ContextSpec,
    columns: list[int],
    fit_metrics: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "response_matrix": [[float(value) for value in row] for row in response_matrix],
        "response_source": RESPONSE_SOURCE,
        "response_axis": RESPONSE_AXIS,
        "response_source_details": {
            "measurement_protocol": "closed_loop_sinusoidal_target_perturbation_v1",
            "candidate_id": candidate_id,
            "controller_name": controller_name,
            "controller_params": dict(controller_params),
            "task_name": context.task_name,
            "config_name": context.config_name,
            "identified_columns": [int(value) for value in columns],
            "amplitude_deg": float(args.amplitude),
            "frequency_hz": float(args.frequency),
            "duration_s": float(args.duration),
            "dt": float(args.dt),
            "ramp_s": float(args.ramp_s),
            "settle_fraction": float(args.settle_fraction),
            "dry_run": bool(args.dry_run),
            "fit_metrics": fit_metrics,
        },
    }


def _identify_candidate_response(
    *,
    robot,
    candidate_id: str,
    controller_name: str,
    controller_params: dict[str, object],
    context: ContextSpec,
    base_target_deg: np.ndarray,
    all_joint_names: list[str],
    active_gripper_cmd: dict[str, float],
    has_object: bool,
    columns: list[int],
    args: argparse.Namespace,
) -> dict[str, object]:
    logger.info(
        "Identifying %s (%s) for %s on columns=%s",
        candidate_id,
        controller_name,
        context.context_id,
        columns,
    )
    if args.dry_run:
        trials = [
            _simulate_column_trial(
                candidate_id=candidate_id,
                column_index=column_index,
                base_target_deg=base_target_deg,
                amplitude_deg=args.amplitude,
                frequency_hz=args.frequency,
                duration_s=args.duration,
                dt=args.dt,
                ramp_s=args.ramp_s,
            )
            for column_index in columns
        ]
    else:
        ir = make_openarm_dual_arm_ir()
        M_obj = _make_object_inertia(context.task_name)
        controller = make_controller(
            controller_name,
            ir,
            N_PER_ARM,
            "openarm",
            M_obj=M_obj,
            **controller_params,
        )
        trials = [
            _run_column_probe(
                robot,
                controller,
                column_index=column_index,
                all_joint_names=all_joint_names,
                base_target_deg=base_target_deg,
                active_gripper_cmd=active_gripper_cmd,
                has_object=has_object,
                object_mass_kg=float(TASK_OBJECTS[context.task_name]["mass"]),
                amplitude_deg=args.amplitude,
                frequency_hz=args.frequency,
                duration_s=args.duration,
                dt=args.dt,
                ramp_s=args.ramp_s,
            )
            for column_index in columns
        ]

    response, fit_metrics = _response_matrix_from_column_trials(
        trials,
        n_total=N_TOTAL,
        base_target_deg=base_target_deg,
        settle_fraction=args.settle_fraction,
    )
    return _matrix_payload(
        response_matrix=response,
        candidate_id=candidate_id,
        controller_name=controller_name,
        controller_params=controller_params,
        context=context,
        columns=columns,
        fit_metrics=fit_metrics,
        args=args,
    )


def _identify_context(
    *,
    context: ContextSpec,
    candidate_specs: dict[str, tuple[str, dict[str, object]]],
    columns: list[int],
    args: argparse.Namespace,
) -> dict[str, object]:
    _, _, all_joint_names = _joint_name_lists(N_PER_ARM)
    base_target_deg = _base_target_deg(context.config_name, all_joint_names)
    arm_hold_cmd = {
        f"{joint_name}.pos": float(base_target_deg[index])
        for index, joint_name in enumerate(all_joint_names)
    }

    robot = None
    active_gripper_cmd: dict[str, float] = {}
    gripper_targets = (-65.0, 0.0)
    has_object = False
    try:
        if not args.dry_run:
            robot = _make_robot(args)
            robot.connect()
            _prepare_robot_start(
                robot,
                base_target_deg=base_target_deg,
                all_joint_names=all_joint_names,
            )
            has_object, active_gripper_cmd, gripper_targets = _prepare_object_if_needed(
                robot,
                object_mass_kg=float(TASK_OBJECTS[context.task_name]["mass"]),
                arm_hold_cmd=arm_hold_cmd,
            )
            _sync_base_target_from_arm_hold_cmd(base_target_deg, all_joint_names, arm_hold_cmd)
            if has_object:
                is_heavy_object = float(TASK_OBJECTS[context.task_name]["mass"]) >= HEAVY_OBJECT_MASS_KG
                hold_kp, hold_kd = _openarm_object_hold_gains(is_heavy_object=is_heavy_object)
                _adapt_contact_settled_passive_joints(robot, arm_hold_cmd)
                _stabilize_arm_pose_if_needed(
                    robot,
                    arm_hold_cmd=arm_hold_cmd,
                    active_gripper_cmd=active_gripper_cmd,
                    custom_kp=hold_kp,
                    custom_kd=hold_kd,
                )

        return {
            candidate_id: _identify_candidate_response(
                robot=robot,
                candidate_id=candidate_id,
                controller_name=controller_name,
                controller_params=controller_params,
                context=context,
                base_target_deg=base_target_deg,
                all_joint_names=all_joint_names,
                active_gripper_cmd=active_gripper_cmd,
                has_object=has_object,
                columns=columns,
                args=args,
            )
            for candidate_id, (controller_name, controller_params) in candidate_specs.items()
        }
    finally:
        if robot is not None:
            try:
                active_gripper_cmd = _release_object_if_needed(
                    robot,
                    has_object=has_object,
                    object_mass_kg=float(TASK_OBJECTS[context.task_name]["mass"]),
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
                logger.info("Cooling/parking after %s for %.1fs.", context.context_id, POST_OBJECT_COOLDOWN_S)
                time.sleep(POST_OBJECT_COOLDOWN_S)
                slow_move(
                    robot,
                    final_cmd,
                    duration_s=3.0,
                    custom_kp=GRIPPER_HOLD_KP,
                    custom_kd=GRIPPER_HOLD_KD,
                )
            except Exception:
                logger.exception("Failed to park robot cleanly after hardware response identification")
            try:
                robot.disconnect()
            except Exception:
                logger.exception("Failed to disconnect robot cleanly")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify Matrix B hardware response models on OpenArm")
    parser.add_argument("--left-port", default="can3")
    parser.add_argument("--right-port", default="can2")
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Repeatable task:config context, e.g. bar_only:bar_b",
    )
    parser.add_argument("--task", choices=sorted(TASK_OBJECTS), default="bar_loaded")
    parser.add_argument("--config", choices=sorted(COORDINATION_CONFIGS), default="bar_b")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        choices=sorted(DEFAULT_CANDIDATES),
        help="Repeatable Matrix B candidate id. Defaults to all four first-round candidates.",
    )
    parser.add_argument(
        "--controller-params-input",
        default=None,
        help="Optional JSON mapping candidate_id -> controller_params overrides.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        type=int,
        default=list(range(N_TOTAL)),
        help="Full joint indices to identify. Defaults to all 14 columns.",
    )
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=0.35)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--ramp-s", type=float, default=1.0)
    parser.add_argument("--settle-fraction", type=float, default=0.25)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contexts = [_parse_context(raw) for raw in args.context]
    if not contexts:
        contexts = [ContextSpec(task_name=str(args.task), config_name=str(args.config))]
    columns = sorted({int(value) for value in args.columns})
    invalid_columns = [value for value in columns if value < 0 or value >= N_TOTAL]
    if invalid_columns:
        raise ValueError(f"columns out of range 0..{N_TOTAL - 1}: {invalid_columns}")
    candidate_ids = list(args.candidate_id) or list(DEFAULT_CANDIDATES)
    candidate_specs = _candidate_specs(
        candidate_ids,
        controller_params_override=_parse_controller_params(args.controller_params_input),
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot": "openarm",
        "response_source": RESPONSE_SOURCE,
        "response_axis": RESPONSE_AXIS,
        "context_ids": [context.context_id for context in contexts],
        "candidate_ids": candidate_ids,
        "identified_columns": columns,
        "dry_run": bool(args.dry_run),
    }
    for context in contexts:
        payload[context.context_id] = _identify_context(
            context=context,
            candidate_specs=candidate_specs,
            columns=columns,
            args=args,
        )

    output_path = Path(args.output) if args.output else (
        DEFAULT_OUTPUT / f"openarm_hardware_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Saved Matrix B hardware response models to %s", output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
