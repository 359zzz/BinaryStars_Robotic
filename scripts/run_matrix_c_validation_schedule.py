#!/usr/bin/env python3
"""Build or execute a Matrix C v1 real-robot validation schedule.

This is a thin orchestration layer. It consumes Matrix C decision-support
output and translates it into:

1. candidate validation priority
2. manual reconfiguration checkpoints
3. existing BinaryStars_Real runner invocations
4. an optional execution log

It deliberately does NOT:
- reinterpret raw Matrix B channels
- treat signed pair margins as left/right winner semantics
- claim that the QPU autonomously discovers candidate families
- claim that Matrix C emits a globally optimal morphology
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "matrix_c_validation"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def validate_matrix_c_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "matrix_c_decision_v1":
        raise ValueError("Expected matrix_c_decision_v1 payload")

    if payload.get("source_family_type") != "morphology":
        raise ValueError("This orchestrator currently supports morphology Matrix C payloads only")

    risk_guards = payload.get("risk_guards", {})
    if risk_guards.get("pair_margin_sign_policy") != "distinguishability_only":
        raise ValueError("Matrix C payload is not risk-safe: pair margin sign policy must be distinguishability_only")
    if risk_guards.get("uses_raw_channels") not in (False, None):
        raise ValueError("Matrix C payload is not risk-safe: raw channels must not be used")
    if risk_guards.get("claims_autonomous_family_discovery"):
        raise ValueError("Matrix C payload is not risk-safe: autonomous family discovery claim is not allowed")
    if risk_guards.get("claims_qpu_emits_globally_optimal_candidate"):
        raise ValueError("Matrix C payload is not risk-safe: globally optimal candidate claim is not allowed")
    if risk_guards.get("decision_semantics") != "priority_shortlist_validation_schedule_only":
        raise ValueError("Matrix C payload must remain decision-support only")


def build_candidate_index(graph_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not graph_payload:
        return {}
    return {
        node["candidate_id"]: node
        for node in graph_payload.get("nodes", [])
        if "candidate_id" in node
    }


def infer_baseline_candidate_id(
    candidate_index: dict[str, dict[str, Any]],
    candidate_priority: list[dict[str, Any]],
) -> str | None:
    for candidate_id, node in candidate_index.items():
        if node.get("variant_metadata", {}).get("kind") == "baseline":
            return candidate_id
    for entry in candidate_priority:
        candidate_id = entry.get("candidate_id", "")
        if "baseline" in candidate_id:
            return candidate_id
    return None


def local_joint_indices_from_probe_pair(probe_pair: list[int], n_per_arm: int = 7) -> list[int]:
    local_joint_indices: list[int] = []
    for probe_idx in probe_pair:
        local_idx = int(probe_idx) % n_per_arm
        if local_idx not in local_joint_indices:
            local_joint_indices.append(local_idx)
    return sorted(local_joint_indices)


def make_command_step(
    *,
    step_id: str,
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
    description: str,
    candidate_node: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = candidate_node.get("variant_metadata", {}) if candidate_node else {}
    operator_checks = [
        "Confirm the physical morphology variant is installed and documented.",
        "Confirm calibration / joint zero is still valid after reconfiguration.",
        "Confirm no new self-collision or cable interference has been introduced.",
    ]
    return {
        "step_id": step_id,
        "step_kind": "manual_reconfiguration",
        "protocol_family": "manual_reconfiguration",
        "description": description,
        "variant_metadata": metadata,
        "operator_checks": operator_checks,
        "status": "planned",
    }


def candidate_validation_steps(
    *,
    candidate_id: str,
    candidate_node: dict[str, Any] | None,
    args: argparse.Namespace,
    local_probe_joints: list[int],
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    if candidate_node and candidate_node.get("variant_metadata", {}).get("kind") != "baseline":
        steps.append(
            make_manual_step(
                step_id=f"{candidate_id}_manual_reconfigure",
                description=candidate_node.get(
                    "description",
                    f"Manually install candidate variant {candidate_id} before validation.",
                ),
                candidate_node=candidate_node,
            )
        )

    steps.append(
        make_command_step(
            step_id=f"{candidate_id}_preflight_single_arm",
            protocol_family="preflight_single_arm",
            description="Single-arm preflight before morphology validation.",
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
            step_id=f"{candidate_id}_preflight_dual_arm",
            protocol_family="preflight_dual_arm",
            description="Dual-arm preflight before downstream coordination / cross-arm checks.",
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
                    step_id=f"{candidate_id}_coupling_{config_name}_j{local_joint_idx}",
                    protocol_family="single_arm_coupling",
                    description=(
                        "Probe-focused single-arm coupling check aligned with Matrix B distal probe indices."
                    ),
                    command=command,
                    output_dir=coupling_dir,
                    supports_dry_run=True,
                    notes=[
                        "Uses existing run_openarm_coupling.py runner.",
                        "This is a morphology-sensitive physical check, not a direct winner declaration.",
                    ],
                )
            )

    lemma_dir = candidate_dir / "lemma3"
    steps.append(
        make_command_step(
            step_id=f"{candidate_id}_dual_arm_lemma3",
            protocol_family="dual_arm_null_control",
            description="Cross-arm null/control check using the existing Lemma 3 runner.",
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
                "Hardware-only step: current runner has no dry-run mode.",
                "Treat this as a physical null/control anchor, not as Matrix C preference evidence.",
            ],
        )
    )

    if args.include_coordination:
        coordination_dir = candidate_dir / "coordination"
        coordination_reps = "1" if args.profile == "compact" else "3"
        coordination_duration = "6.0" if args.profile == "compact" else "10.0"
        command = [
            args.python,
            "scripts/run_coordination.py",
            "--robot",
            args.robot,
            "--task",
            args.coordination_task,
            "--controller",
            args.coordination_controller,
            "--config",
            args.coordination_config,
            "--reps",
            coordination_reps,
            "--duration",
            coordination_duration,
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
                step_id=f"{candidate_id}_coordination_downstream",
                protocol_family="downstream_coordination",
                description=(
                    "Optional downstream coordination surrogate. This is useful for closing the B->C->D loop,"
                    " but it is still only a downstream physical validation protocol."
                ),
                command=command,
                output_dir=coordination_dir,
                supports_dry_run=True,
                notes=[
                    "Uses an available real-robot coordination surrogate rather than a literal morphology swap benchmark.",
                    "Keep interpretation downstream: better performance here does not retroactively turn Matrix C into a winner oracle.",
                ],
            )
        )

    return steps


def build_manifest(
    decision_payload: dict[str, Any],
    graph_payload: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_matrix_c_payload(decision_payload)

    candidate_index = build_candidate_index(graph_payload)
    priority_entries = decision_payload.get("candidate_priority", [])
    priority_by_id = {entry["candidate_id"]: entry for entry in priority_entries}

    shortlist_ids = list(decision_payload.get("validation_shortlist", {}).get("candidate_ids", []))
    if args.top_k is not None:
        shortlist_ids = shortlist_ids[: args.top_k]

    baseline_candidate_id = None
    if args.include_baseline_reference:
        baseline_candidate_id = infer_baseline_candidate_id(candidate_index, priority_entries)

    selected_candidate_ids: list[str] = []
    if baseline_candidate_id:
        selected_candidate_ids.append(baseline_candidate_id)
    for candidate_id in shortlist_ids:
        if candidate_id not in selected_candidate_ids:
            selected_candidate_ids.append(candidate_id)

    selected_edges = []
    for edge in decision_payload.get("validation_schedule", []):
        edge_candidate_ids = edge.get("candidate_ids", [])
        if all(candidate_id in selected_candidate_ids for candidate_id in edge_candidate_ids):
            selected_edges.append(edge)

    local_probe_joint_indices: list[int] = []
    for edge in selected_edges:
        for local_idx in local_joint_indices_from_probe_pair(edge.get("probe_pair", [])):
            if local_idx not in local_probe_joint_indices:
                local_probe_joint_indices.append(local_idx)
    if not local_probe_joint_indices:
        local_probe_joint_indices = [6]

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"matrix_c_validation_{run_stamp}"
    run_dir = Path(args.output_dir) / run_id

    candidate_validations = []
    for order, candidate_id in enumerate(selected_candidate_ids, start=1):
        candidate_node = candidate_index.get(candidate_id)
        priority_entry = priority_by_id.get(candidate_id, {})
        candidate_dir = run_dir / candidate_id
        role = "baseline_reference" if candidate_id == baseline_candidate_id else "validation_shortlist"
        steps = candidate_validation_steps(
            candidate_id=candidate_id,
            candidate_node=candidate_node,
            args=args,
            local_probe_joints=local_probe_joint_indices,
            candidate_dir=candidate_dir,
        )
        candidate_validations.append(
            {
                "candidate_id": candidate_id,
                "role": role,
                "order": order,
                "priority_rank": priority_entry.get("priority_rank"),
                "priority_score": priority_entry.get("priority_score"),
                "label": (candidate_node or {}).get("label", candidate_id),
                "description": (candidate_node or {}).get("description"),
                "variant_metadata": (candidate_node or {}).get("variant_metadata", {}),
                "supporting_edge_ids": priority_entry.get("supporting_edge_ids", []),
                "steps": steps,
            }
        )

    return {
        "schema": "matrix_c_validation_schedule_v1",
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "source_matrix_c_path": str(Path(args.matrix_c).resolve()),
        "source_matrix_b_graph_path": str(Path(args.matrix_b_graph).resolve()) if args.matrix_b_graph else None,
        "source_graph_id": decision_payload.get("source_graph_id"),
        "source_family_type": decision_payload.get("source_family_type"),
        "source_context": decision_payload.get("source_context", {}),
        "source_matrix_c_readiness": decision_payload.get("source_matrix_c_readiness"),
        "risk_guards": decision_payload.get("risk_guards", {}),
        "execution_policy": {
            "robot": args.robot,
            "profile": args.profile,
            "dry_run_requested": args.dry_run,
            "execute_requested": args.execute,
            "include_baseline_reference": args.include_baseline_reference,
            "include_coordination": args.include_coordination,
            "coordination_task": args.coordination_task,
            "coordination_controller": args.coordination_controller,
            "coordination_config": args.coordination_config,
            "python": args.python,
        },
        "decision_semantics": {
            "mode": decision_payload.get("decision_mode"),
            "shortlist_policy": decision_payload.get("validation_shortlist", {}).get("policy"),
            "notes": [
                "This schedule treats Matrix C as a validation-priority module only.",
                "No step in this manifest should be interpreted as a QPU-declared winner.",
                "Manual morphology changes remain human-in-the-loop and must be explicitly audited.",
            ],
        },
        "selected_candidates": selected_candidate_ids,
        "selected_edges": selected_edges,
        "local_probe_joint_indices": local_probe_joint_indices,
        "candidate_validations": candidate_validations,
        "controls_summary": decision_payload.get("controls_summary", {}),
        "warnings": decision_payload.get("warnings", []),
    }


def execute_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    execution_log: dict[str, Any] = {
        "schema": "matrix_c_validation_execution_v1",
        "executed_at": utc_now_iso(),
        "run_id": manifest.get("run_id"),
        "dry_run_requested": args.dry_run,
        "steps": [],
    }

    for candidate in manifest.get("candidate_validations", []):
        for step in candidate.get("steps", []):
            log_row: dict[str, Any] = {
                "candidate_id": candidate.get("candidate_id"),
                "step_id": step.get("step_id"),
                "step_kind": step.get("step_kind"),
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
    parser = argparse.ArgumentParser(description="Matrix C v1 real-robot validation schedule orchestrator")
    parser.add_argument("--matrix-c", required=True, help="Path to matrix_c_decision_v1.json")
    parser.add_argument("--matrix-b-graph", help="Optional path to matrix_b_family_graph_v1.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for schedule manifests")
    parser.add_argument("--robot", choices=["openarm"], default="openarm")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for downstream runner invocations")
    parser.add_argument("--port", default="can0", help="Single-arm OpenArm port")
    parser.add_argument("--side", default="right", help="Single-arm OpenArm side")
    parser.add_argument("--left-port", default="can1", help="Dual-arm left port")
    parser.add_argument("--right-port", default="can0", help="Dual-arm right port")
    parser.add_argument("--top-k", type=int, help="Override Matrix C shortlist top-k")
    parser.add_argument("--profile", choices=["compact", "full"], default="compact")
    parser.add_argument("--dry-run", action="store_true", help="Prefer dry-run capable downstream runners when possible")
    parser.add_argument("--execute", action="store_true", help="Execute the generated command steps after manifest creation")
    parser.add_argument("--auto-ack-manual", action="store_true", help="Do not pause for manual reconfiguration checkpoints")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop execution after the first failed runner")
    parser.add_argument(
        "--include-baseline-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include baseline candidate as a physical reference stage",
    )
    parser.add_argument("--include-coordination", action="store_true", help="Append a downstream coordination surrogate to each candidate schedule")
    parser.add_argument("--coordination-task", default="bar_only", choices=["independent", "bar_only", "bar_loaded"])
    parser.add_argument("--coordination-controller", default="c_coupled", choices=["decoupled", "j_coupled", "c_coupled", "s_adaptive"])
    parser.add_argument("--coordination-config", default="bar_b", choices=["bar_mid", "bar_a", "bar_b", "bar_c"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_c_path = Path(args.matrix_c)
    matrix_b_graph_path = Path(args.matrix_b_graph) if args.matrix_b_graph else None

    decision_payload = load_json(matrix_c_path)
    graph_payload = load_json(matrix_b_graph_path) if matrix_b_graph_path else None
    manifest = build_manifest(decision_payload, graph_payload, args)

    manifest_path = Path(args.output_dir) / manifest["run_id"] / "validation_schedule_manifest.json"
    dump_json(manifest_path, manifest)
    print(f"[MATRIX_C_ORCH] Wrote manifest to {manifest_path}")

    if not args.execute:
        return 0

    execution_log = execute_manifest(manifest, args)
    execution_path = Path(args.output_dir) / manifest["run_id"] / "validation_schedule_execution.json"
    dump_json(execution_path, execution_log)
    print(f"[MATRIX_C_ORCH] Wrote execution log to {execution_path}")

    failed_steps = [step for step in execution_log.get("steps", []) if step.get("status") == "failed"]
    return 1 if failed_steps else 0


if __name__ == "__main__":
    raise SystemExit(main())
