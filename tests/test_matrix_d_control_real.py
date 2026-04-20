from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _bootstrap_repo() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_module():
    _bootstrap_repo()
    return _load_script_module(
        "run_matrix_d_real_validation",
        _repo_root() / "scripts" / "run_matrix_d_real_validation.py",
    )


def _report_module():
    _bootstrap_repo()
    from bsreal.experiment import matrix_d_real_report

    return matrix_d_real_report


def _control_graph() -> dict[str, object]:
    return {
        "nodes": [
            {
                "candidate_id": "decoupled_ref",
                "label": "Decoupled Reference",
                "controller_name": "decoupled",
                "controller_params": {},
            },
            {
                "candidate_id": "j_coupled_eng",
                "label": "J-Coupled Engineering Anchor",
                "controller_name": "j_coupled",
                "controller_params": {"kp_comp": 2.0, "kd_comp": 0.1, "alpha_pos": 0.3},
            },
            {
                "candidate_id": "c_coupled_cross",
                "label": "C-Coupled Cross-Arm",
                "controller_name": "c_coupled",
                "controller_params": {"kp_comp": 2.0, "kd_comp": 0.1, "alpha_pos": 0.3},
            },
            {
                "candidate_id": "s_adaptive_entropy",
                "label": "S-Adaptive Entropy",
                "controller_name": "s_adaptive",
                "controller_params": {
                    "kp_comp": 2.0,
                    "kd_comp": 0.1,
                    "alpha_pos": 0.3,
                    "s_threshold": 0.3,
                    "recompute_every": 10,
                },
            },
        ],
        "edges": [
            {
                "edge_id": "decoupled__c",
                "probe_pair": [2, 10],
            }
        ],
    }


def _control_c2() -> dict[str, object]:
    return {
        "schema": "matrix_c2_real_v1",
        "source_graph_id": "control_graph",
        "source_family_type": "control",
        "source_context": {"robot": "OpenArm v10", "task": "bar_loaded"},
        "source_matrix_c_readiness": "strict_handoff",
        "routing_mode": "control_real_robot_validation_routing_v1",
        "reference_candidate_id": "decoupled_ref",
        "engineering_anchor_candidate_id": "j_coupled_eng",
        "engineering_anchor_validate_candidates": ["j_coupled_eng"],
        "primary_validate_candidates": ["c_coupled_cross"],
        "secondary_validate_candidates": ["s_adaptive_entropy"],
        "optional_or_deferred_candidates": [],
        "abstain_candidates": [],
        "pairwise_validation_order": [
            {
                "pair": ["decoupled_ref", "j_coupled_eng"],
                "route_role": "reference_anchor_comparison",
                "route_rank": 1,
                "evidence_edge_id": "decoupled__c",
            }
        ],
        "required_control_checks": [
            "baseline_repeat_before_controller_swap",
            "post_swap_preflight",
            "matched_probe_coupling_check",
            "downstream_task_battery",
        ],
        "real_validation_budget_weights": {"policy": "demo", "weights": {"decoupled_ref": 0.2}},
        "risk_guards": {
            "pair_margin_sign_policy": "distinguishability_only",
            "uses_raw_channels": False,
            "baseline_reserved_as_reference": True,
            "engineering_anchor_reserved": True,
            "decision_semantics": "real_robot_validation_routing_not_winner",
        },
        "candidate_routes": [
            {
                "candidate_id": "decoupled_ref",
                "route_status": "reference_baseline",
                "route_reason": "baseline_reference",
                "priority_rank": 2,
                "priority_score": 2.1,
                "supporting_edge_ids": ["decoupled__c"],
                "feasibility": {},
            },
            {
                "candidate_id": "j_coupled_eng",
                "route_status": "engineering_anchor_validate",
                "route_reason": "reserved_engineering_anchor",
                "priority_rank": 3,
                "priority_score": 1.8,
                "supporting_edge_ids": ["anchor__c"],
                "feasibility": {},
            },
            {
                "candidate_id": "c_coupled_cross",
                "route_status": "primary_validate",
                "route_reason": "shortlisted_primary_validation_target",
                "priority_rank": 1,
                "priority_score": 2.4,
                "supporting_edge_ids": ["decoupled__c", "anchor__c"],
                "feasibility": {},
            },
            {
                "candidate_id": "s_adaptive_entropy",
                "route_status": "secondary_validate",
                "route_reason": "shortlisted_secondary_validation_target",
                "priority_rank": 4,
                "priority_score": 1.6,
                "supporting_edge_ids": ["decoupled__s"],
                "feasibility": {},
            },
        ],
        "warnings": [],
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        c2_real_input=str(tmp_path / "c2.json"),
        graph_input=str(tmp_path / "graph.json"),
        output_dir=str(tmp_path),
        robot="openarm",
        python=sys.executable,
        port="can0",
        side="right",
        left_port="can1",
        right_port="can0",
        profile="compact",
        dry_run=True,
        execute=False,
        auto_ack_manual=False,
        stop_on_error=False,
        include_optional_candidates=False,
        include_coordination=True,
        coordination_controller="c_coupled",
        coordination_config="bar_b",
        coordination_tasks=["bar_loaded"],
        coordination_reps=1,
        coordination_duration=5.0,
        control_probe_task="bar_loaded",
        control_probe_amplitude=3.0,
        control_probe_frequency=0.5,
        control_probe_duration=5.0,
        skip_control_probe=False,
        strict_preflight_sync_gate=True,
        preflight_sync_threshold_deg=1.0,
        annotation_input=None,
        grasp_mode=None,
        hardware_confound=[],
        operator_note=[],
        auto_build_report=False,
        aggregate_report_input=[],
        aggregate_output=None,
        min_clean_runs_for_freeze=2,
    )


def test_build_manifest_supports_control_family(tmp_path: Path) -> None:
    module = _runner_module()
    manifest = module.build_manifest(_control_c2(), _control_graph(), _args(tmp_path))

    assert manifest["source_family_type"] == "control"
    assert manifest["engineering_anchor_candidate_id"] == "j_coupled_eng"
    assert manifest["probe_pairs"] == [[2, 10]]
    assert [
        row["candidate_id"] for row in manifest["real_validation_execution_order"]
    ] == [
        "decoupled_ref",
        "j_coupled_eng",
        "c_coupled_cross",
        "s_adaptive_entropy",
    ]

    anchor_row = next(
        row for row in manifest["candidate_validations"] if row["candidate_id"] == "j_coupled_eng"
    )
    assert anchor_row["controller_name"] == "j_coupled"
    assert all(step["step_kind"] == "command" for step in anchor_row["steps"])
    assert any(step["protocol_family"] == "control_probe" for step in anchor_row["steps"])
    assert any(
        step["protocol_family"] == "preflight_dual_arm"
        and "--require-sync-ok" in step["command"]
        and "--summary-output" in step["command"]
        for step in anchor_row["steps"]
    )
    assert any(
        step["protocol_family"] == "downstream_coordination"
        and "j_coupled" in step["command"]
        for step in anchor_row["steps"]
    )


def test_build_report_supports_control_probe_and_anchor(tmp_path: Path) -> None:
    report_module = _report_module()
    run_dir = tmp_path / "matrix_d_real_20260418_000000"
    run_dir.mkdir(parents=True)

    manifest = {
        "schema": "matrix_d_real_manifest_v1",
        "run_id": run_dir.name,
        "source_graph_id": "control_graph",
        "source_family_type": "control",
        "source_matrix_c_readiness": "strict_handoff",
        "reference_candidate_id": "decoupled_ref",
        "engineering_anchor_candidate_id": "j_coupled_eng",
        "execution_policy": {
            "coordination_tasks": ["bar_loaded"],
            "coordination_reps": 1,
        },
        "candidate_validations": [
            {
                "candidate_id": "decoupled_ref",
                "route_status": "reference_baseline",
                "scheduled_for_execution": True,
                "scheduling_reason": "baseline_reference",
            },
            {
                "candidate_id": "j_coupled_eng",
                "route_status": "engineering_anchor_validate",
                "scheduled_for_execution": True,
                "scheduling_reason": "engineering_anchor_reference",
            },
            {
                "candidate_id": "c_coupled_cross",
                "route_status": "primary_validate",
                "scheduled_for_execution": True,
                "scheduling_reason": "primary_validation_target",
            },
            {
                "candidate_id": "s_adaptive_entropy",
                "route_status": "secondary_validate",
                "scheduled_for_execution": True,
                "scheduling_reason": "secondary_validation_target",
            },
        ],
    }
    execution = {
        "schema": "matrix_d_real_execution_v1",
        "steps": [
            {
                "candidate_id": "decoupled_ref",
                "stage": "D1_probe_matched_physical_check",
                "status": "completed",
            }
        ],
    }
    (run_dir / "matrix_d_real_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "matrix_d_real_execution.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )

    per_candidate = {
        "decoupled_ref": 1.50,
        "j_coupled_eng": 1.20,
        "c_coupled_cross": 1.00,
        "s_adaptive_entropy": 1.10,
    }
    for candidate_id, rmse in per_candidate.items():
        preflight_dir = run_dir / candidate_id / "preflight"
        preflight_dir.mkdir(parents=True)
        (preflight_dir / "preflight_dual_arm.json").write_text(
            json.dumps(
                {
                    "schema": "coordination_preflight_summary_v1",
                    "status": "passed",
                    "sync_check": {"sync_error_deg": 0.2, "sync_ok": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        probe_dir = run_dir / candidate_id / "control_probe_pair_2_10"
        probe_dir.mkdir(parents=True)
        (probe_dir / "control_probe_pair_2_10.json").write_text(
            json.dumps(
                {
                    "schema": "matrix_d_control_probe_v1",
                    "timestamp_utc": "2026-04-18T00:00:00+00:00",
                    "task": "bar_loaded",
                    "controller": candidate_id,
                    "probe_pair": [2, 10],
                    "aggregate": {
                        "direction_count": 2,
                        "mean_full_body_rmse_deg": rmse / 2.0,
                        "mean_opposite_probe_hold_rmse_deg": rmse / 3.0,
                        "max_opposite_probe_peak_abs_error_deg": rmse / 1.5,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        coordination_dir = run_dir / candidate_id / "coordination_bar_loaded"
        coordination_dir.mkdir(parents=True)
        (coordination_dir / "bar_loaded.json").write_text(
            json.dumps(
                {
                    "task": "bar_loaded",
                    "controller": candidate_id,
                    "config": "bar_b",
                    "rep": 0,
                    "rmse_right": rmse,
                    "rmse_left": rmse,
                    "rmse_total": rmse,
                    "s_rho_l": 1.0,
                    "j_cross_max": 0.5,
                    "n_samples": 100,
                    "contact_settled_passive_joint_targets": (
                        {"left_joint_5.pos": -6.0} if candidate_id == "c_coupled_cross" else {}
                    ),
                    "contact_settled_passive_joint_nominal_targets": (
                        {"left_joint_5.pos": -75.0} if candidate_id == "c_coupled_cross" else {}
                    ),
                    "contact_settled_passive_joint_errors_deg": (
                        {"left_joint_5.pos": 69.0} if candidate_id == "c_coupled_cross" else {}
                    ),
                    "pretrial_pose_stabilization_events": [
                        {"stage": "post_grasp_settle", "attempt_count": 1, "recovered": True}
                    ],
                    "timestamp_utc": "2026-04-18T00:00:00+00:00",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    report = report_module.build_matrix_d_real_report(
        run_dir,
        annotation={"grasp_mode": "true_grasp"},
    )

    assert report["engineering_anchor_candidate_id"] == "j_coupled_eng"
    assert len(report["d1_summary"]["preflight_runs"]) == 4
    assert len(report["d1_summary"]["control_probe_runs"]) == 4
    assert report["d1_summary"]["control_probe_pairwise_vs_reference"]
    assert report["statistical_report"]["c2_alignment"]["status"] == "supports_c2_routing"
    assert report["statistical_report"]["d1_probe_alignment"]["status"] in {
        "supports_primary_probe_alignment",
        "supports_primary_probe_over_reference_only",
    }
    assert (
        report["statistical_report"]["engineering_anchor_alignment"]["status"]
        == "primary_beats_engineering_anchor"
    )
    assert report["statistical_report"]["pairwise_vs_engineering_anchor"]
    assert report["engineering_adjustments"]["contact_settled_trial_count"] == 1


def test_build_manifest_accepts_matrix_c_after_e_control_payload(tmp_path: Path) -> None:
    module = _runner_module()
    c_after_e = _control_c2()
    c_after_e["schema"] = "matrix_c_after_e_control_v1"
    c_after_e["risk_guards"]["decision_semantics"] = "confirmatory_validation_routing_after_matrix_e"

    manifest = module.build_manifest(c_after_e, _control_graph(), _args(tmp_path))

    assert manifest["source_family_type"] == "control"
    assert manifest["candidate_validations"][0]["candidate_id"] == "decoupled_ref"
