"""Dual-arm coordination experiment framework.

Runs 4 tasks x 4 controllers x 5 configs x N reps and records RMSE + coupling metrics.
"""

from __future__ import annotations

import json
import math
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bsreal.dynamics.models import DynamicsIR
from bsreal.dynamics.mass_matrix import compute_mass_matrix
from bsreal.dynamics.coupling import normalized_coupling_matrix
from bsreal.dynamics.effective_mass import (
    compute_M_eff_for_dual_arm, make_object_spatial_inertia,
)
from bsreal.experiment.controllers import BaseController, make_controller
from bsreal.experiment.trajectory import (
    TASK_OBJECTS, COORDINATION_CONFIGS,
    generate_bimanual_trajectory, get_start_positions_deg,
)
from bsreal.experiment.safety import slow_move, emergency_freeze, SafetyError

logger = logging.getLogger(__name__)

GRIPPER_LATCH_KP = {"gripper": 10.0}
GRIPPER_LATCH_KD = {"gripper": 0.25}
GRIPPER_HOLD_KP = {"gripper": 8.0}
GRIPPER_HOLD_KD = {"gripper": 0.2}


@dataclass
class CoordinationConfig:
    task_name: str           # "independent", "box_lift", "barbell_lift", "rod_rotation"
    object_mass_kg: float
    object_geometry: str
    object_dims: tuple
    controller_name: str     # "decoupled", "j_coupled", "c_coupled", "s_adaptive"
    config_name: str         # "home", "elbow_up", etc.
    n_repetitions: int = 3
    duration_s: float = 10.0
    dt: float = 0.01
    precompensate: bool = False   # offline trajectory pre-compensation
    precomp_alpha: float = 0.3    # pre-compensation gain


@dataclass
class CoordinationResult:
    config: CoordinationConfig
    rep: int
    timestamps: list[float] = field(default_factory=list)
    q_target_deg: list[list[float]] = field(default_factory=list)
    q_actual_deg: list[list[float]] = field(default_factory=list)
    torques_ff: list[list[float]] = field(default_factory=list)
    rmse_right: float = 0.0
    rmse_left: float = 0.0
    rmse_total: float = 0.0
    s_rho_l: float = 0.0
    j_cross_max: float = 0.0
    n_samples: int = 0


def _openarm_gripper_targets(robot) -> tuple[float, float]:
    if hasattr(robot, "right_arm") and hasattr(robot, "left_arm"):
        right_limits = robot.right_arm.config.joint_limits.get("gripper", (-65.0, 0.0))
        left_limits = robot.left_arm.config.joint_limits.get("gripper", (-65.0, 0.0))
        return min(float(right_limits[0]), float(left_limits[0])), max(
            float(right_limits[1]), float(left_limits[1])
        )
    if hasattr(robot, "config"):
        limits = robot.config.joint_limits.get("gripper", (-65.0, 0.0))
        return float(limits[0]), float(limits[1])
    return -65.0, 0.0


def _gripper_targets(robot, robot_type: str) -> tuple[float, float]:
    """Return (open, closed) gripper targets in LeRobot degrees.

    For OpenArm, infer the real working range from the LeRobot config so the
    experiment code stays aligned with calibration / side-specific limits.
    """
    if robot_type == "piper":
        return 80.0, 0.0
    return _openarm_gripper_targets(robot)


def _dual_gripper_cmd(target: float) -> dict[str, float]:
    return {"right_gripper.pos": target, "left_gripper.pos": target}


def _current_dual_gripper_cmd(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {
        "right_gripper.pos": obs.get("right_gripper.pos", 0.0),
        "left_gripper.pos": obs.get("left_gripper.pos", 0.0),
    }


def _send_action(robot, cmd: dict[str, float], *, custom_kp=None, custom_kd=None) -> None:
    if custom_kp is not None or custom_kd is not None:
        robot.send_action(cmd, custom_kp=custom_kp, custom_kd=custom_kd)
    else:
        robot.send_action(cmd)


def _send_gripper_repeated(
    robot,
    target: float,
    *,
    duration_s: float = 1.0,
    dt: float = 0.05,
    custom_kp=None,
    custom_kd=None,
) -> dict[str, float]:
    """Send a dual-gripper command repeatedly so the hold state latches."""
    cmd = _dual_gripper_cmd(target)
    n_steps = max(int(duration_s / dt), 1)
    for _ in range(n_steps):
        _send_action(robot, cmd, custom_kp=custom_kp, custom_kd=custom_kd)
        time.sleep(dt)
    _send_action(robot, cmd, custom_kp=custom_kp, custom_kd=custom_kd)
    return cmd


def _hold_gripper_target_until_enter(
    robot,
    target: float,
    prompt: str,
    *,
    dt: float = 0.05,
    custom_kp=None,
    custom_kd=None,
) -> dict[str, float]:
    """Keep sending the same dual-gripper target while waiting for manual input."""
    cmd = _dual_gripper_cmd(target)
    stop_event = threading.Event()

    def _worker():
        while not stop_event.is_set():
            _send_action(robot, cmd, custom_kp=custom_kp, custom_kd=custom_kd)
            time.sleep(dt)
        _send_action(robot, cmd, custom_kp=custom_kp, custom_kd=custom_kd)

    worker = threading.Thread(target=_worker, name="coordination-gripper-hold", daemon=True)
    worker.start()
    try:
        input(prompt)
    finally:
        stop_event.set()
        worker.join(timeout=max(0.2, 4 * dt))
    return cmd


def _compute_rmse(target: np.ndarray, actual: np.ndarray, joint_range: range) -> float:
    """Compute RMSE in degrees over specified joint range."""
    diff = target[:, list(joint_range)] - actual[:, list(joint_range)]
    return float(np.sqrt(np.mean(diff**2)))


def _compute_coupling_metrics(
    ir: DynamicsIR, q_rad: np.ndarray, n_per_arm: int,
    M_obj: np.ndarray | None,
) -> tuple[float, float]:
    """Compute S(rho_L) and max cross-arm |J_ij| at given configuration."""
    if M_obj is not None and not np.allclose(M_obj, 0.0):
        M_eff = compute_M_eff_for_dual_arm(ir, q_rad, n_per_arm, M_obj)
    else:
        M_eff = compute_mass_matrix(ir, q_rad)

    J_mat = normalized_coupling_matrix(M_eff)
    n = n_per_arm
    cross = J_mat[:n, n:]
    j_cross_max = float(np.max(np.abs(cross)))

    # Entropy from singular values of cross-arm block
    sv = np.linalg.svd(cross, compute_uv=False)
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0, j_cross_max
    sv2 = sv**2
    sv2 = sv2 / sv2.sum()
    S = -float(np.sum(sv2 * np.log2(sv2 + 1e-15)))
    return S, j_cross_max


def run_coordination_trial(
    robot,
    controller: BaseController,
    ir: DynamicsIR,
    coord_config: CoordinationConfig,
    rep: int,
    joint_names_right: list[str],
    joint_names_left: list[str],
    dry_run: bool = False,
) -> CoordinationResult:
    """Execute one coordination trial.

    Steps:
    1. slow_move to start pose
    2. If object: close grippers
    3. 100 Hz control loop, recording q_target, q_actual, torques
    4. Compute RMSE
    5. Open grippers, return
    """
    result = CoordinationResult(config=coord_config, rep=rep)
    all_joint_names = joint_names_right + joint_names_left
    n_per_arm = len(joint_names_right)
    n_total = 2 * n_per_arm

    # Generate trajectory
    timestamps, q_right_deg, q_left_deg = generate_bimanual_trajectory(
        coord_config.config_name, coord_config.duration_s, coord_config.dt,
    )
    q_target_all_deg = np.hstack([q_right_deg, q_left_deg])

    # Object spatial inertia (compute early — needed for pre-compensation)
    obj = TASK_OBJECTS[coord_config.task_name]
    M_obj = None
    if obj["mass"] > 0:
        M_obj = make_object_spatial_inertia(
            obj["mass"], obj["geometry"], obj["dims"],
        )

    # Offline trajectory pre-compensation (for position-controlled robots)
    if coord_config.precompensate and M_obj is not None:
        from bsreal.experiment.trajectory_precompensation import (
            precompensate_trajectory, precompensation_summary,
        )
        q_target_all_deg = precompensate_trajectory(
            q_target_all_deg, ir, n_per_arm, M_obj,
            alpha=coord_config.precomp_alpha,
        )
        summary = precompensation_summary(
            np.hstack([q_right_deg, q_left_deg]), q_target_all_deg, n_per_arm,
        )
        logger.info(
            f"  Pre-compensation applied: alpha={coord_config.precomp_alpha}, "
            f"max_corr={summary['max_correction_right_deg']:.2f}° (R), "
            f"{summary['max_correction_left_deg']:.2f}° (L)"
        )

    if dry_run:
        # Simulate with zero tracking error + noise
        q_actual_all = q_target_all_deg + np.random.randn(*q_target_all_deg.shape) * 0.5
        result.timestamps = timestamps.tolist()
        result.q_target_deg = q_target_all_deg.tolist()
        result.q_actual_deg = q_actual_all.tolist()
        result.torques_ff = np.zeros_like(q_target_all_deg).tolist()
        result.n_samples = len(timestamps)
        result.rmse_right = _compute_rmse(
            q_target_all_deg, q_actual_all, range(0, n_per_arm),
        )
        result.rmse_left = _compute_rmse(
            q_target_all_deg, q_actual_all, range(n_per_arm, n_total),
        )
        result.rmse_total = _compute_rmse(
            q_target_all_deg, q_actual_all, range(0, n_total),
        )
        # Coupling metrics at start config
        q_start_rad = np.deg2rad(q_target_all_deg[0])
        result.s_rho_l, result.j_cross_max = _compute_coupling_metrics(
            ir, q_start_rad, n_per_arm, M_obj,
        )
        return result

    # 1. Slow move to start. Keep grippers at their current command during
    # repositioning so gripper motors do not drop out of the command stream.
    start_pos = get_start_positions_deg(coord_config.config_name)
    start_cmd = {f"{jn}.pos": start_pos.get(jn, 0.0) for jn in all_joint_names}
    start_cmd.update(_current_dual_gripper_cmd(robot))
    if controller.robot_type == "openarm":
        slow_move(
            robot,
            start_cmd,
            duration_s=3.0,
            custom_kp=GRIPPER_HOLD_KP,
            custom_kd=GRIPPER_HOLD_KD,
        )
    else:
        slow_move(robot, start_cmd, duration_s=3.0)
    time.sleep(1.0)

    # 2. Object placement
    has_object = coord_config.object_mass_kg > 0
    gripper_open, gripper_close = _gripper_targets(robot, controller.robot_type)
    active_gripper_cmd = _current_dual_gripper_cmd(robot)
    gripper_latch_kp = GRIPPER_LATCH_KP if controller.robot_type == "openarm" else None
    gripper_latch_kd = GRIPPER_LATCH_KD if controller.robot_type == "openarm" else None
    gripper_hold_kp = GRIPPER_HOLD_KP if controller.robot_type == "openarm" else None
    gripper_hold_kd = GRIPPER_HOLD_KD if controller.robot_type == "openarm" else None
    if has_object:
        _send_gripper_repeated(
            robot,
            gripper_open,
            duration_s=0.5,
            custom_kp=gripper_latch_kp,
            custom_kd=gripper_latch_kd,
        )
        _hold_gripper_target_until_enter(
            robot,
            gripper_open,
            (
                f"\n>>> Grippers are being held open. Place the bar "
                f"({coord_config.object_mass_kg} kg) between both grippers, "
                f"then press ENTER..."
            ),
            custom_kp=gripper_hold_kp,
            custom_kd=gripper_hold_kd,
        )
        print("  Grippers closing...")
        _send_gripper_repeated(
            robot,
            gripper_close,
            duration_s=1.25,
            custom_kp=gripper_latch_kp,
            custom_kd=gripper_latch_kd,
        )
        active_gripper_cmd = _send_gripper_repeated(
            robot,
            gripper_close,
            duration_s=0.6,
            custom_kp=gripper_hold_kp,
            custom_kd=gripper_hold_kd,
        )

    # 3. Control loop
    logger.info(
        f"Trial: {coord_config.task_name}/{coord_config.controller_name}/"
        f"{coord_config.config_name} rep={rep}"
    )

    t0 = time.monotonic()
    step = 0
    n_errors = 0

    try:
        while step < len(timestamps):
            t = time.monotonic() - t0
            if t >= coord_config.duration_s:
                break

            # Find the closest target step
            step = min(int(t / coord_config.dt), len(timestamps) - 1)
            q_tgt_deg = q_target_all_deg[step]
            q_tgt_rad = np.deg2rad(q_tgt_deg)

            # Read observation
            obs = robot.get_observation()
            q_cur_deg = np.array([obs.get(f"{jn}.pos", 0.0) for jn in all_joint_names])
            q_cur_rad = np.deg2rad(q_cur_deg)
            qdot_cur = np.array([obs.get(f"{jn}.vel", 0.0) for jn in all_joint_names])
            qdot_cur_rad = np.deg2rad(qdot_cur)  # Piper has no vel, defaults to 0

            # Compute control action
            qdot_tgt = np.zeros(n_total)  # zero velocity target for now
            tau_ff = controller.compute_action(
                t, q_cur_rad, qdot_cur_rad, q_tgt_rad, qdot_tgt,
            )

            # Build command
            cmd = {}
            for j, jn in enumerate(all_joint_names):
                cmd[f"{jn}.pos"] = float(q_tgt_deg[j])
                if controller.robot_type == "openarm":
                    cmd[f"{jn}.tau_ff"] = float(tau_ff[j])
                else:
                    # Piper: apply position compensation
                    cmd[f"{jn}.pos"] = float(q_tgt_deg[j] + math.degrees(tau_ff[j]))
            if has_object:
                cmd.update(active_gripper_cmd)

            _send_action(
                robot,
                cmd,
                custom_kp=gripper_hold_kp if has_object else None,
                custom_kd=gripper_hold_kd if has_object else None,
            )

            # Record
            result.timestamps.append(t)
            result.q_target_deg.append(q_tgt_deg.tolist())
            result.q_actual_deg.append(q_cur_deg.tolist())
            result.torques_ff.append(tau_ff.tolist())

            # Safety check — relaxed for settling and for position-only robots
            try:
                from bsreal.experiment.safety import check_position_error
                cur_dict = {jn: q_cur_deg[j] for j, jn in enumerate(all_joint_names)}
                tgt_dict = {jn: q_tgt_deg[j] for j, jn in enumerate(all_joint_names)}
                if t < 3.0:
                    max_err = 45.0
                elif controller.robot_type == "piper":
                    max_err = 45.0  # position control + j5 calibration offset
                else:
                    max_err = 15.0
                check_position_error(cur_dict, tgt_dict, max_error=max_err)
                n_errors = 0
            except SafetyError as e:
                n_errors += 1
                if n_errors >= 5:
                    logger.error(f"Safety: {e}")
                    emergency_freeze(robot)
                    raise

            # Pace
            sleep_time = coord_config.dt - (time.monotonic() - t0 - t)
            if sleep_time > 0:
                time.sleep(sleep_time)

            step += 1

    except SafetyError:
        raise
    except KeyboardInterrupt:
        logger.info("Trial interrupted")
    finally:
        if has_object:
            _hold_gripper_target_until_enter(
                robot,
                gripper_close,
                "\n>>> Support or remove the bar, then press ENTER to release the grippers...",
                custom_kp=gripper_hold_kp,
                custom_kd=gripper_hold_kd,
            )
            active_gripper_cmd = _send_gripper_repeated(
                robot,
                gripper_open,
                duration_s=0.8,
                custom_kp=gripper_latch_kp,
                custom_kd=gripper_latch_kd,
            )
            time.sleep(0.3)

        # Return only after the grippers have been unloaded/opened.
        start_cmd.update(active_gripper_cmd)
        if controller.robot_type == "openarm":
            slow_move(
                robot,
                start_cmd,
                duration_s=3.0,
                custom_kp=GRIPPER_HOLD_KP,
                custom_kd=GRIPPER_HOLD_KD,
            )
        else:
            slow_move(robot, start_cmd, duration_s=3.0)

    # 4. Compute RMSE
    if len(result.q_target_deg) > 10:
        tgt = np.array(result.q_target_deg)
        act = np.array(result.q_actual_deg)
        result.rmse_right = _compute_rmse(tgt, act, range(0, n_per_arm))
        result.rmse_left = _compute_rmse(tgt, act, range(n_per_arm, n_total))
        result.rmse_total = _compute_rmse(tgt, act, range(0, n_total))

    result.n_samples = len(result.timestamps)

    # Coupling metrics at start configuration
    q_start_rad = np.deg2rad(q_target_all_deg[0])
    result.s_rho_l, result.j_cross_max = _compute_coupling_metrics(
        ir, q_start_rad, n_per_arm, M_obj,
    )

    logger.info(
        f"  RMSE: R={result.rmse_right:.3f} L={result.rmse_left:.3f} "
        f"total={result.rmse_total:.3f} deg | S={result.s_rho_l:.3f} "
        f"J_cross_max={result.j_cross_max:.3f}"
    )
    return result


def run_coordination_suite(
    robot,
    ir: DynamicsIR,
    output_dir: str | Path,
    n_per_arm: int = 7,
    robot_type: str = "openarm",
    tasks: list[str] | None = None,
    controllers: list[str] | None = None,
    configs: list[str] | None = None,
    n_reps: int = 3,
    dry_run: bool = False,
    precompensate: bool = False,
    precomp_alpha: float = 0.3,
) -> dict:
    """Batch execution: tasks x controllers x configs x reps.

    Returns summary dict and saves per-trial JSON files.
    """
    if tasks is None:
        tasks = list(TASK_OBJECTS.keys())
    if controllers is None:
        controllers = ["decoupled", "j_coupled", "c_coupled", "s_adaptive"]
    if configs is None:
        configs = list(COORDINATION_CONFIGS.keys())

    if robot_type == "openarm":
        jn_right = [f"right_joint_{i}" for i in range(1, n_per_arm + 1)]
        jn_left = [f"left_joint_{i}" for i in range(1, n_per_arm + 1)]
    else:
        jn_right = [f"right_joint_{i}" for i in range(1, n_per_arm + 1)]
        jn_left = [f"left_joint_{i}" for i in range(1, n_per_arm + 1)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(tasks) * len(controllers) * len(configs) * n_reps
    done = 0

    for task_name in tasks:
        obj = TASK_OBJECTS[task_name]
        M_obj = None
        if obj["mass"] > 0:
            M_obj = make_object_spatial_inertia(
                obj["mass"], obj["geometry"], obj["dims"],
            )

        for ctrl_name in controllers:
            ctrl = make_controller(
                ctrl_name, ir, n_per_arm, robot_type, M_obj=M_obj,
            )

            for config_name in configs:
                for rep in range(n_reps):
                    done += 1
                    logger.info(f"[{done}/{total}] {task_name}/{ctrl_name}/{config_name} rep={rep}")

                    cc = CoordinationConfig(
                        task_name=task_name,
                        object_mass_kg=obj["mass"],
                        object_geometry=obj["geometry"],
                        object_dims=obj.get("dims", ()),
                        controller_name=ctrl_name,
                        config_name=config_name,
                        n_repetitions=n_reps,
                        precompensate=precompensate,
                        precomp_alpha=precomp_alpha,
                    )

                    result = run_coordination_trial(
                        robot, ctrl, ir, cc, rep,
                        jn_right, jn_left, dry_run=dry_run,
                    )
                    all_results.append(result)

                    # Save individual trial
                    trial_file = (
                        out_dir / f"{task_name}_{ctrl_name}_{config_name}_rep{rep}.json"
                    )
                    _save_result(result, trial_file)

                    if not dry_run:
                        time.sleep(1.0)

    # Save summary
    summary = _build_summary(all_results)
    summary_file = out_dir / "coordination_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_file}")

    return summary


def _save_result(result: CoordinationResult, path: Path):
    """Save a single trial result to JSON."""
    data = {
        "task": result.config.task_name,
        "controller": result.config.controller_name,
        "config": result.config.config_name,
        "object_mass_kg": result.config.object_mass_kg,
        "rep": result.rep,
        "n_samples": result.n_samples,
        "rmse_right": result.rmse_right,
        "rmse_left": result.rmse_left,
        "rmse_total": result.rmse_total,
        "s_rho_l": result.s_rho_l,
        "j_cross_max": result.j_cross_max,
        "timestamps": result.timestamps,
        "q_target_deg": result.q_target_deg,
        "q_actual_deg": result.q_actual_deg,
        "torques_ff": result.torques_ff,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _build_summary(results: list[CoordinationResult]) -> dict:
    """Build summary statistics from all trials."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r.config.task_name, r.config.controller_name, r.config.config_name)
        groups[key].append(r)

    rows = []
    for (task, ctrl, config), trials in sorted(groups.items()):
        rmses = [t.rmse_total for t in trials]
        rows.append({
            "task": task,
            "controller": ctrl,
            "config": config,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_min": float(np.min(rmses)),
            "rmse_max": float(np.max(rmses)),
            "s_rho_l": trials[0].s_rho_l,
            "j_cross_max": trials[0].j_cross_max,
            "n_reps": len(trials),
        })

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials": len(results),
        "rows": rows,
    }
