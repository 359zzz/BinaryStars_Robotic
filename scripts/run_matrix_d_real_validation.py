#!/usr/bin/env python3
"""Build or execute a Matrix D real-robot validation schedule from Matrix C2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "matrix_d_real"
DEFAULT_CONTROL_CONTROLLER_MAP = {
    "decoupled_ref": ("decoupled", {}),
    "j_coupled_eng": ("j_coupled", {"kp_comp": 2.0, "kd_comp": 0.1, "alpha_pos": 0.3}),
    "c_coupled_cross": ("c_coupled", {"kp_comp": 2.0, "kd_comp": 0.1, "alpha_pos": 0.3}),
    "s_adaptive_entropy": (
        "s_adaptive",
        {
            "kp_comp": 2.0,
            "kd_comp": 0.1,
            "alpha_pos": 0.3,
            "s_threshold": 0.3,
            "recompute_every": 10,
        },
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def validate_matrix_c2_real(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "matrix_c2_real_v1":
        raise ValueError("Expected matrix_c2_real_v1 payload")
    family_type = str(payload.get("source_family_type"))
    if family_type not in {"morphology", "control"}:
        raise ValueError("This D_real_v1 runner supports only morphology or control family payloads")

    risk_guards = payload.get("risk_guards", {})
    if risk_guards.get("pair_margin_sign_policy") != "distinguishability_only":
        raise ValueError("Matrix C2 real payload must preserve distinguishability_only semantics")
    if risk_guards.get("uses_raw_channels") not in (False, None):
        raise ValueError("Matrix C2 real payload must not use raw channels")
    if not risk_guards.get("baseline_reserved_as_reference", False):
        raise ValueError("Matrix C2 real payload must reserve a reference baseline")
    if risk_guards.get("decision_semantics") != "real_robot_validation_routing_not_winner":
        raise ValueError("Matrix C2 real payload must remain routing-only")
    if family_type == "control" and not risk_guards.get("engineering_anchor_reserved", False):
        raise ValueError("Control Matrix C2 real payload must reserve an engineering anchor")


def build_candidate_index(graph_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not graph_payload:
        return {}
    return {
        node["candidate_id"]: node
        for node in graph_payload.get("nodes", [])
        if "candidate_id" in node
    }


def local_joint_indices_from_probe_pair(probe_pair: list[int], n_per_arm: int = 7) -> list[int]:
    local_joint_indices: list[int] = []
    for probe_idx in probe_pair:
        local_idx = int(probe_idx) % n_per_arm
        if local_idx not in local_joint_indices:
            local_joint_indices.append(local_idx)
    return sorted(local_joint_indices)


def infer_probe_pairs(
    c2_payload: dict[str, Any],
    graph_payload: dict[str, Any] | None,
) -> list[list[int]]:
    graph_edges = {}
    if graph_payload:
        for edge in graph_payload.get("edges", []):
            edge_id = edge.get("edge_id")
            if edge_id:
                graph_edges[str(edge_id)] = edge

    probe_pairs: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for row in c2_payload.get("pairwise_validation_order", []):
        edge_id = row.get("evidence_edge_id")
        if not edge_id:
            continue
        edge = graph_edges.get(str(edge_id), {})
        probe_pair = edge.get("probe_pair")
        if not isinstance(probe_pair, list) or len(probe_pair) != 2:
            continue
        key = (int(probe_pair[0]), int(probe_pair[1]))
        if key in seen:
            continue
        seen.add(key)
        probe_pairs.append([key[0], key[1]])

    return probe_pairs or [[6, 13]]


def infer_probe_joints(
    c2_payload: dict[str, Any],
    graph_payload: dict[str, Any] | None,
) -> list[int]:
    local_joint_indices: list[int] = []
    for probe_pair in infer_probe_pairs(c2_payload, graph_payload):
        for local_idx in local_joint_indices_from_probe_pair(probe_pair):
            if local_idx not in local_joint_indices:
                local_joint_indices.append(local_idx)

    return local_joint_indices or [6]


def make_command_step(
    *,
    step_id: str,
    stage: str,
    protocol_family: str,
    description: str,
    command: list[str],
    output_dir: Path | None,
    supports_dry_run: bool,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "step_kind": "command",
        "stage": stage,
        "protocol_family": protocol_family,
        "description": description,
        "command": command,
        "cwd": str(REPO_ROOT),
        "output_dir": str(output_dir) if output_dir else None,
        "supports_dry_run": supports_dry_run,
        "notes": notes or [],
        "status": "planned",
    }


def make_manual_step(
    *,
    step_id: str,
    stage: str,
    description: str,
    candidate_node: dict[str, Any] | None,
    feasibility_row: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = candidate_node.get("variant_metadata", {}) if candidate_node else {}
    notes = list((feasibility_row or {}).get("notes", []))
    operator_checks = [
        "Confirm the physical candidate is installed and documented.",
        "Confirm calibration / joint zero is still valid after reconfiguration.",
        "Confirm no new self-collision, cable interference, or unsafe workspace limits were introduced.",
        "Record mass / geometry / hardware deltas before proceeding.",
    ]
    return {
        "step_id": step_id,
        "step_kind": "manual_reconfiguration",
        "stage": stage,
        "protocol_family": "manual_reconfiguration_audit",
        "description": description,
        "variant_metadata": metadata,
        "feasibility": feasibility_row or {},
        "operator_checks": operator_checks,
        "notes": notes,
        "status": "planned",
    }


def control_candidate_spec(
    candidate_id: str,
    candidate_node: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if candidate_node:
        controller_name = candidate_node.get("controller_name")
        controller_params = candidate_node.get("controller_params", {})
        if isinstance(controller_name, str) and controller_name:
            if not isinstance(controller_params, dict):
                controller_params = {}
            return controller_name, dict(controller_params)

    fallback_name, fallback_params = DEFAULT_CONTROL_CONTROLLER_MAP.get(
        candidate_id,
        ("decoupled", {}),
    )
    return fallback_name, dict(fallback_params)


def candidate_validation_steps(
    *,
    candidate_id: str,
    role: str,
    candidate_node: dict[str, Any] | None,
    feasibility_row: dict[str, Any] | None,
    family_type: str,
    probe_pairs: list[list[int]],
    local_probe_joints: list[int],
    args: argparse.Namespace,
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    if family_type == "control":
        return control_candidate_validation_steps(
            candidate_id=candidate_id,
            role=role,
            candidate_node=candidate_node,
            feasibility_row=feasibility_row,
            probe_pairs=probe_pairs,
            args=args,
            candidate_dir=candidate_dir,
        )

    steps: list[dict[str, Any]] = []
    is_reference = role == "reference_baseline"

    if not is_reference:
        steps.append(
            make_manual_step(
                step_id=f"{candidate_id}_d0_reconfiguration_audit",
                stage="D0_reconfiguration_audit",
                description=(candidate_node or {}).get(
                    "description",
                    f"Install and audit candidate {candidate_id} before D_real validation.",
                ),
                candidate_node=candidate_node,
                feasibility_row=feasibility_row,
            )
        )

    steps.append(
        make_command_step(
            step_id=f"{candidate_id}_d1_preflight_single_arm",
            stage="D1_probe_matched_physical_check",
            protocol_family="preflight_single_arm",
            description="Single-arm preflight before probe-matched physical validation.",
            command=[
                args.python,
                "scripts/preflight.py",
                "--robot",
                args.robot,
                "--port",
                args.port,
                "--side",
                args.side,
            ],
            output_dir=None,
            supports_dry_run=False,
        )
    )

    steps.append(
        make_command_step(
            step_id=f"{candidate_id}_d1_preflight_dual_arm",
            stage="D1_probe_matched_physical_check",
            protocol_family="preflight_dual_arm",
            description="Dual-arm preflight before null/control and downstream validation.",
            command=[
                args.python,
                "scripts/preflight_coordination.py",
                "--robot",
                args.robot,
                "--config",
                args.coordination_config,
                "--left-port",
                args.left_port,
                "--right-port",
                args.right_port,
            ],
            output_dir=None,
            supports_dry_run=False,
        )
    )

    coupling_configs = ["wrist_twist", "full_pose"] if args.profile == "compact" else ["wrist_twist", "full_pose", "shoulder_elbow"]
    for config_name in coupling_configs:
        for local_joint_idx in local_probe_joints:
            coupling_dir = candidate_dir / f"coupling_{config_name}_j{local_joint_idx}"
            command = [
                args.python,
                "scripts/run_openarm_coupling.py",
                "--config",
                config_name,
                "--joint-start",
                str(local_joint_idx),
                "--joint-end",
                str(local_joint_idx),
                "--port",
                args.port,
                "--side",
                args.side,
                "--output-dir",
                str(coupling_dir),
            ]
            if args.dry_run:
                command.append("--dry-run")
            steps.append(
                make_command_step(
                    step_id=f"{candidate_id}_d1_coupling_{config_name}_j{local_joint_idx}",
                    stage="D1_probe_matched_physical_check",
                    protocol_family="single_arm_coupling",
                    description="Probe-matched single-arm coupling validation aligned with the Matrix B family signal.",
                    command=command,
                    output_dir=coupling_dir,
                    supports_dry_run=True,
                    notes=[
                        "This checks whether the candidate changes local dynamics along the QPU-matched probe direction.",
                        "Interpret downstream only; do not treat this as a winner declaration.",
                    ],
                )
            )

    lemma_dir = candidate_dir / "lemma3"
    steps.append(
        make_command_step(
            step_id=f"{candidate_id}_d1_dual_arm_lemma3",
            stage="D1_probe_matched_physical_check",
            protocol_family="dual_arm_null_control",
            description="Cross-arm null/control verification using the existing Lemma 3 runner.",
            command=[
                args.python,
                "scripts/run_dual_arm_lemma3.py",
                "--left-port",
                args.left_port,
                "--right-port",
                args.right_port,
                "--joints",
                *[str(local_joint_idx) for local_joint_idx in local_probe_joints],
                "--output-dir",
                str(lemma_dir),
            ],
            output_dir=lemma_dir,
            supports_dry_run=False,
            notes=[
                "Treat this as a physical control anchor, not as direct preference evidence.",
            ],
        )
    )

    if args.include_coordination:
        for task_name in args.coordination_tasks:
            coordination_dir = candidate_dir / f"coordination_{task_name}"
            command = [
                args.python,
                "scripts/run_coordination.py",
                "--robot",
                args.robot,
                "--task",
                task_name,
                "--controller",
                args.coordination_controller,
                "--config",
                args.coordination_config,
                "--reps",
                str(args.coordination_reps),
                "--duration",
                str(args.coordination_duration),
                "--left-port",
                args.left_port,
                "--right-port",
                args.right_port,
                "--output-dir",
                str(coordination_dir),
            ]
            if args.dry_run:
                command.append("--dry-run")
            steps.append(
                make_command_step(
                    step_id=f"{candidate_id}_d2_coordination_{task_name}",
                    stage="D2_downstream_task_battery",
                    protocol_family="downstream_coordination",
                    description=(
                        f"Downstream coordination validation under task={task_name}."
                    ),
                    command=command,
                    output_dir=coordination_dir,
                    supports_dry_run=True,
                    notes=[
                        "This is the downstream real-robot battery for C2-guided routing.",
                        "Candidate performance is interpreted relative to the baseline reference and safety envelope.",
                    ],
                )
            )

    return steps


def control_candidate_validation_steps(
    *,
    candidate_id: str,
    role: str,
    candidate_node: dict[str, Any] | None,
    feasibility_row: dict[str, Any] | None,
    probe_pairs: list[list[int]],
    args: argparse.Namespace,
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    controller_name, controller_params = control_candidate_spec(candidate_id, candidate_node)
    controller_params_json = json.dumps(controller_params, separators=(",", ":"), ensure_ascii=False)

    steps.append(
        make_command_step(
            step_id=f"{candidate_id}_d1_preflight_dual_arm",
            stage="D1_probe_matched_physical_check",
            protocol_family="preflight_dual_arm",
            description="Dual-arm preflight before controller-sensitive D1 probe and downstream validation.",
            command=[
                args.python,
                "scripts/preflight_coordination.py",
                "--robot",
                args.robot,
                "--config",
                args.coordination_config,
                "--left-port",
                args.left_port,
                "--right-port",
                args.right_port,
            ],
            output_dir=None,
            supports_dry_run=False,
        )
    )

    if not args.skip_control_probe:
        for probe_pair in probe_pairs:
            probe_dir = candidate_dir / f"control_probe_pair_{probe_pair[0]}_{probe_pair[1]}"
            command = [
                args.python,
                "scripts/run_control_probe.py",
                "--robot",
                args.robot,
                "--left-port",
                args.left_port,
                "--right-port",
                args.right_port,
                "--task",
                args.control_probe_task,
                "--config",
                args.coordination_config,
                "--controller",
                controller_name,
                "--controller-params-json",
                controller_params_json,
                "--probe-pair",
                str(probe_pair[0]),
                str(probe_pair[1]),
                "--amplitude",
                str(args.control_probe_amplitude),
                "--frequency",
                str(args.control_probe_frequency),
                "--duration",
                str(args.control_probe_duration),
                "--output-dir",
                str(probe_dir),
            ]
            if args.dry_run:
                command.append("--dry-run")
            steps.append(
                make_command_step(
                    step_id=f"{candidate_id}_d1_control_probe_pair_{probe_pair[0]}_{probe_pair[1]}",
                    stage="D1_probe_matched_physical_check",
                    protocol_family="control_probe",
                    description="Controller-sensitive dual-arm probe aligned with the Matrix B probe pair.",
                    command=command,
                    output_dir=probe_dir,
                    supports_dry_run=True,
                    notes=[
                        "This is the control-family D1 check and should vary with the deployed controller candidate.",
                        "Interpret it as matched downstream validation support, not as a QPU winner declaration.",
                    ],
                )
            )

    if args.include_coordination:
        for task_name in args.coordination_tasks:
            coordination_dir = candidate_dir / f"coordination_{task_name}"
            command = [
                args.python,
                "scripts/run_coordination.py",
                "--robot",
                args.robot,
                "--task",
                task_name,
                "--controller",
                controller_name,
                "--controller-params-json",
                controller_params_json,
                "--config",
                args.coordination_config,
                "--reps",
                str(args.coordination_reps),
                "--duration",
                str(args.coordination_duration),
                "--left-port",
                args.left_port,
                "--right-port",
                args.right_port,
                "--output-dir",
                str(coordination_dir),
            ]
            if args.dry_run:
                command.append("--dry-run")
            steps.append(
                make_command_step(
                    step_id=f"{candidate_id}_d2_coordination_{task_name}",
                    stage="D2_downstream_task_battery",
                    protocol_family="downstream_coordination",
                    description=f"Same-robot downstream coordination validation under task={task_name}.",
                    command=command,
                    output_dir=coordination_dir,
                    supports_dry_run=True,
                    notes=[
                        f"Controller candidate: {controller_name}",
                        "This is the real downstream battery for control-family routing on a fixed robot body.",
                    ],
                )
            )

    return steps


def scheduling_decision(
    route_status: str,
    *,
    include_optional_candidates: bool,
) -> tuple[bool, str]:
    if route_status == "reference_baseline":
        return True, "baseline_reference"
    if route_status == "engineering_anchor_validate":
        return True, "engineering_anchor_reference"
    if route_status == "primary_validate":
        return True, "primary_validation_target"
    if route_status == "secondary_validate":
        return True, "secondary_validation_target"
    if route_status == "optional_validate":
        if include_optional_candidates:
            return True, "optional_candidate_explicitly_enabled"
        return False, "optional_candidate_not_enabled"
    if route_status in {"deferred_not_available", "deferred_not_ready"}:
        return False, "deferred_candidate_not_ready_for_execution"
    if route_status.startswith("abstain_"):
        return False, "abstained_candidate"
    return False, "unscheduled_route_status"


def build_manifest(
    c2_payload: dict[str, Any],
    graph_payload: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_matrix_c2_real(c2_payload)
    candidate_index = build_candidate_index(graph_payload)
    family_type = str(c2_payload.get("source_family_type"))
    probe_pairs = infer_probe_pairs(c2_payload, graph_payload)
    local_probe_joints = infer_probe_joints(c2_payload, graph_payload)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"matrix_d_real_{run_stamp}"
    run_dir = Path(args.output_dir) / run_id

    route_rows = c2_payload.get("candidate_routes", [])
    route_by_id = {
        str(row.get("candidate_id")): row
        for row in route_rows
        if row.get("candidate_id") is not None
    }

    ordered_candidate_ids = []
    baseline_id = str(c2_payload.get("reference_candidate_id"))
    if baseline_id:
        ordered_candidate_ids.append(baseline_id)
    for key in [
        "engineering_anchor_validate_candidates",
        "primary_validate_candidates",
        "secondary_validate_candidates",
        "optional_or_deferred_candidates",
        "abstain_candidates",
    ]:
        for candidate_id in c2_payload.get(key, []):
            candidate_id = str(candidate_id)
            if candidate_id not in ordered_candidate_ids:
                ordered_candidate_ids.append(candidate_id)

    candidate_validations = []
    for candidate_id in ordered_candidate_ids:
        route_row = dict(route_by_id.get(candidate_id, {}))
        role = route_row.get("route_status", "unknown")
        feasibility_row = dict(route_row.get("feasibility", {}))
        candidate_node = candidate_index.get(candidate_id)
        controller_name, controller_params = (
            control_candidate_spec(candidate_id, candidate_node)
            if family_type == "control"
            else (None, {})
        )
        candidate_dir = run_dir / candidate_id
        scheduled_for_execution, scheduling_reason = scheduling_decision(
            str(role),
            include_optional_candidates=args.include_optional_candidates,
        )
        steps = []
        if scheduled_for_execution:
            steps = candidate_validation_steps(
                candidate_id=candidate_id,
                role=str(role),
                candidate_node=candidate_node,
                feasibility_row=feasibility_row,
                family_type=family_type,
                probe_pairs=probe_pairs,
                local_probe_joints=local_probe_joints,
                args=args,
                candidate_dir=candidate_dir,
            )
        candidate_validations.append(
            {
                "candidate_id": candidate_id,
                "route_status": role,
                "route_reason": route_row.get("route_reason"),
                "priority_rank": route_row.get("priority_rank"),
                "priority_score": route_row.get("priority_score"),
                "supporting_edge_ids": route_row.get("supporting_edge_ids", []),
                "feasibility": feasibility_row,
                "controller_name": controller_name,
                "controller_params": controller_params,
                "scheduled_for_execution": scheduled_for_execution,
                "scheduling_reason": scheduling_reason,
                "label": (candidate_node or {}).get("label", candidate_id),
                "description": (candidate_node or {}).get("description"),
                "variant_metadata": (candidate_node or {}).get("variant_metadata", {}),
                "steps": steps,
            }
        )

    execution_order = [
        {
            "candidate_id": row["candidate_id"],
            "route_status": row["route_status"],
            "execution_rank": rank,
        }
        for rank, row in enumerate(
            [row for row in candidate_validations if row.get("scheduled_for_execution")],
            start=1,
        )
    ]

    return {
        "schema": "matrix_d_real_manifest_v1",
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "source_matrix_c2_real_path": str(Path(args.c2_real_input).resolve()),
        "source_graph_path": str(Path(args.graph_input).resolve()) if args.graph_input else None,
        "source_graph_id": c2_payload.get("source_graph_id"),
        "source_family_type": family_type,
        "source_context": c2_payload.get("source_context", {}),
        "source_matrix_c_readiness": c2_payload.get("source_matrix_c_readiness"),
        "risk_guards": c2_payload.get("risk_guards", {}),
        "execution_policy": {
            "robot": args.robot,
            "profile": args.profile,
            "dry_run_requested": args.dry_run,
            "execute_requested": args.execute,
            "python": args.python,
            "coordination_controller": args.coordination_controller,
            "coordination_config": args.coordination_config,
            "coordination_tasks": list(args.coordination_tasks),
            "coordination_reps": args.coordination_reps,
            "coordination_duration": args.coordination_duration,
            "include_optional_candidates": args.include_optional_candidates,
            "control_probe_task": args.control_probe_task,
            "control_probe_amplitude": args.control_probe_amplitude,
            "control_probe_frequency": args.control_probe_frequency,
            "control_probe_duration": args.control_probe_duration,
            "skip_control_probe": args.skip_control_probe,
        },
        "decision_semantics": {
            "mode": c2_payload.get("routing_mode"),
            "notes": (
                [
                    "D_real_v1 consumes C2 real routing and does not reinterpret QPU pair margins.",
                    "Baseline remains a physical reference rather than a competing winner candidate.",
                    "Only baseline, primary, and secondary candidates are scheduled by default.",
                    "Optional candidates require explicit opt-in before they enter real-robot execution.",
                    "Abstained and deferred candidates are explicitly excluded from real-robot execution until support or readiness improves.",
                ]
                if family_type != "control"
                else [
                    "D_control_real_v1 consumes C2 real routing over same-robot controller candidates.",
                    "Reference baseline and engineering anchor are reserved comparators rather than winner claims.",
                    "D1 uses a controller-sensitive probe aligned with the Matrix B probe pair before D2 downstream tasks.",
                    "Optional candidates require explicit opt-in before they enter real-robot execution.",
                    "Abstained and deferred candidates are explicitly excluded from real-robot execution until support or readiness improves.",
                ]
            ),
        },
        "reference_candidate_id": baseline_id,
        "engineering_anchor_candidate_id": c2_payload.get("engineering_anchor_candidate_id"),
        "probe_pairs": probe_pairs,
        "local_probe_joint_indices": local_probe_joints,
        "pairwise_validation_order": c2_payload.get("pairwise_validation_order", []),
        "real_validation_execution_order": execution_order,
        "required_control_checks": c2_payload.get("required_control_checks", []),
        "real_validation_budget_weights": c2_payload.get("real_validation_budget_weights", {}),
        "candidate_validations": candidate_validations,
        "warnings": c2_payload.get("warnings", []),
    }


def execute_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    execution_log: dict[str, Any] = {
        "schema": "matrix_d_real_execution_v1",
        "executed_at": utc_now_iso(),
        "run_id": manifest.get("run_id"),
        "dry_run_requested": args.dry_run,
        "steps": [],
    }

    for candidate in manifest.get("candidate_validations", []):
        route_status = str(candidate.get("route_status", ""))
        if not candidate.get("scheduled_for_execution", False):
            execution_log["steps"].append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "route_status": route_status,
                    "status": "skipped_not_scheduled_for_execution",
                    "reason": candidate.get("scheduling_reason"),
                }
            )
            continue

        for step in candidate.get("steps", []):
            log_row: dict[str, Any] = {
                "candidate_id": candidate.get("candidate_id"),
                "step_id": step.get("step_id"),
                "step_kind": step.get("step_kind"),
                "stage": step.get("stage"),
                "protocol_family": step.get("protocol_family"),
                "status": "planned",
            }

            if step.get("step_kind") == "manual_reconfiguration":
                if args.auto_ack_manual or args.dry_run:
                    log_row["status"] = "acknowledged"
                else:
                    print("")
                    print(f"[MANUAL] {candidate.get('candidate_id')}: {step.get('description')}")
                    input("Press ENTER after the candidate is installed and safety-checked...")
                    log_row["status"] = "acknowledged"
                execution_log["steps"].append(log_row)
                continue

            if step.get("step_kind") != "command":
                log_row["status"] = "skipped_unknown_step_kind"
                execution_log["steps"].append(log_row)
                continue

            if args.dry_run and not step.get("supports_dry_run", False):
                log_row["status"] = "skipped_hardware_only"
                execution_log["steps"].append(log_row)
                continue

            command = step.get("command", [])
            log_row["command"] = command
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            log_row["returncode"] = completed.returncode
            log_row["status"] = "completed" if completed.returncode == 0 else "failed"
            execution_log["steps"].append(log_row)

            if completed.returncode != 0 and args.stop_on_error:
                return execution_log

    return execution_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix D real-robot validation from Matrix C2 real routing")
    parser.add_argument("--c2-real-input", required=True, help="Path to matrix_c2_real_v1 JSON")
    parser.add_argument("--graph-input", help="Optional path to matrix_b_family_graph_v1 JSON for probe metadata")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for D_real manifests")
    parser.add_argument("--robot", choices=["openarm"], default="openarm")
    parser.add_argument("--python", default=sys.executable, help="Python executable for downstream runner invocations")
    parser.add_argument("--port", default="can0", help="Single-arm OpenArm port")
    parser.add_argument("--side", default="right", help="Single-arm OpenArm side")
    parser.add_argument("--left-port", default="can1", help="Dual-arm left port")
    parser.add_argument("--right-port", default="can0", help="Dual-arm right port")
    parser.add_argument("--profile", choices=["compact", "full"], default="compact")
    parser.add_argument("--dry-run", action="store_true", help="Prefer dry-run capable downstream runners when possible")
    parser.add_argument("--execute", action="store_true", help="Execute the generated command steps after manifest creation")
    parser.add_argument("--auto-ack-manual", action="store_true", help="Do not pause for manual checkpoints")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop execution after first failed runner")
    parser.add_argument("--include-optional-candidates", action="store_true", help="Also schedule optional_validate candidates for real-robot execution")
    parser.add_argument("--include-coordination", action=argparse.BooleanOptionalAction, default=True, help="Include downstream coordination tasks in D2")
    parser.add_argument("--coordination-controller", default="c_coupled", choices=["decoupled", "j_coupled", "c_coupled", "s_adaptive"])
    parser.add_argument("--coordination-config", default="bar_b", choices=["bar_mid", "bar_a", "bar_b", "bar_c"])
    parser.add_argument("--coordination-tasks", nargs="+", default=["bar_only", "bar_loaded"], choices=["independent", "bar_only", "bar_loaded"])
    parser.add_argument("--coordination-reps", type=int, default=3)
    parser.add_argument("--coordination-duration", type=float, default=10.0)
    parser.add_argument("--control-probe-task", default="bar_loaded", choices=["independent", "bar_only", "bar_loaded"])
    parser.add_argument("--control-probe-amplitude", type=float, default=3.0)
    parser.add_argument("--control-probe-frequency", type=float, default=0.5)
    parser.add_argument("--control-probe-duration", type=float, default=10.0)
    parser.add_argument("--skip-control-probe", action="store_true", help="Skip D1 control-probe steps for control-family runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    c2_path = Path(args.c2_real_input)
    graph_path = Path(args.graph_input) if args.graph_input else None

    c2_payload = load_json(c2_path)
    graph_payload = load_json(graph_path) if graph_path else None
    manifest = build_manifest(c2_payload, graph_payload, args)

    manifest_path = Path(args.output_dir) / manifest["run_id"] / "matrix_d_real_manifest.json"
    dump_json(manifest_path, manifest)
    print(f"[MATRIX_D_REAL] Wrote manifest to {manifest_path}")

    if not args.execute:
        return 0

    execution_log = execute_manifest(manifest, args)
    execution_path = Path(args.output_dir) / manifest["run_id"] / "matrix_d_real_execution.json"
    dump_json(execution_path, execution_log)
    print(f"[MATRIX_D_REAL] Wrote execution log to {execution_path}")

    failed_steps = [step for step in execution_log.get("steps", []) if step.get("status") == "failed"]
    return 1 if failed_steps else 0


if __name__ == "__main__":
    raise SystemExit(main())
