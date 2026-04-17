"""Reporting helpers for Matrix D real-robot validation runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


def build_matrix_d_real_report(
    run_dir: str | Path,
    *,
    annotation: Mapping[str, object] | None = None,
    min_coordination_reps: int | None = None,
) -> dict[str, Any]:
    """Build a D3 report for a single Matrix D real-robot run directory."""

    run_path = Path(run_dir).resolve()
    manifest_path = run_path / "matrix_d_real_manifest.json"
    execution_path = run_path / "matrix_d_real_execution.json"
    manifest = _load_json(manifest_path)
    execution = _load_json(execution_path)
    annotation_map = _normalize_annotation(annotation)

    candidate_rows = _candidate_rows(manifest)
    execution_summary = _execution_summary(execution)
    expected_tasks = list(
        _sequence(_mapping(manifest.get("execution_policy", {})).get("coordination_tasks", []))
    )
    expected_reps = int(
        min_coordination_reps
        if min_coordination_reps is not None
        else _mapping(manifest.get("execution_policy", {})).get("coordination_reps", 0)
    )

    d1_summary = _collect_d1_summary(run_path, candidate_rows)
    coordination_trials = _collect_coordination_trials(
        run_path,
        candidate_rows=candidate_rows,
        annotation=annotation_map,
    )
    coordination_stats = _coordination_stats(coordination_trials)
    task_rankings = _task_rankings(coordination_stats, manifest)
    pairwise_vs_reference = _pairwise_vs_reference(coordination_stats, manifest)
    c2_alignment = _c2_alignment(task_rankings, manifest)
    trial_validity_gate = _trial_validity_gate(
        candidate_rows=candidate_rows,
        coordination_trials=coordination_trials,
        execution_summary=execution_summary,
        expected_tasks=expected_tasks,
        expected_reps=expected_reps,
        annotation=annotation_map,
    )
    freeze_policy = _freeze_policy(
        trial_validity_gate=trial_validity_gate,
        c2_alignment=c2_alignment,
        manifest=manifest,
    )
    paper_ready_metrics = _paper_ready_metrics(
        coordination_stats=coordination_stats,
        task_rankings=task_rankings,
        pairwise_vs_reference=pairwise_vs_reference,
        trial_validity_gate=trial_validity_gate,
        manifest=manifest,
    )

    return {
        "schema": "matrix_d_real_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_path),
        "run_id": manifest.get("run_id") or run_path.name,
        "source_manifest_path": str(manifest_path),
        "source_execution_path": str(execution_path),
        "source_graph_id": manifest.get("source_graph_id"),
        "source_family_type": manifest.get("source_family_type"),
        "source_matrix_c_readiness": manifest.get("source_matrix_c_readiness"),
        "reference_candidate_id": manifest.get("reference_candidate_id"),
        "annotation": annotation_map,
        "execution_summary": execution_summary,
        "candidate_routes": [
            {
                "candidate_id": row.get("candidate_id"),
                "route_status": row.get("route_status"),
                "scheduled_for_execution": row.get("scheduled_for_execution"),
                "scheduling_reason": row.get("scheduling_reason"),
            }
            for row in candidate_rows
        ],
        "d1_summary": d1_summary,
        "d2_coordination_trials": coordination_trials,
        "d3_summary": {
            "coordination_stats": coordination_stats,
            "task_rankings": task_rankings,
            "candidate_file_counts": {
                str(row.get("candidate_id")): _candidate_file_count(
                    run_path / str(row.get("candidate_id"))
                )
                for row in candidate_rows
                if row.get("candidate_id") is not None
            },
        },
        "statistical_report": {
            "primary_metric": "rmse_total",
            "direction": "lower_is_better",
            "pairwise_vs_reference": pairwise_vs_reference,
            "c2_alignment": c2_alignment,
        },
        "trial_validity_gate": trial_validity_gate,
        "freeze_policy": freeze_policy,
        "paper_ready_metrics": paper_ready_metrics,
        "notes": [
            "Matrix D report summarizes downstream real-robot evidence only.",
            "It can support or fail to support C2 routing, but it does not retroactively turn Matrix B into a winner oracle.",
            "Freeze decisions must pass both the trial validity gate and the freeze policy.",
        ],
    }


def build_matrix_d_real_cross_run_report(
    reports: Sequence[Mapping[str, object]],
    *,
    min_clean_runs_for_freeze: int = 2,
) -> dict[str, Any]:
    """Aggregate multiple single-run Matrix D reports."""

    normalized_reports = [
        report
        for report in reports
        if isinstance(report, Mapping) and report.get("schema") == "matrix_d_real_report_v1"
    ]
    if not normalized_reports:
        raise ValueError("Cross-run aggregation requires at least one matrix_d_real_report_v1 payload")

    run_rows = []
    freeze_valid_reports = []
    task_candidate_trials: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    winner_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for report in normalized_reports:
        run_id = str(report.get("run_id"))
        validity = _mapping(report.get("trial_validity_gate", {}))
        freeze = _mapping(report.get("freeze_policy", {}))
        alignment = _mapping(_mapping(report.get("statistical_report", {})).get("c2_alignment", {}))
        run_rows.append(
            {
                "run_id": run_id,
                "run_dir": report.get("run_dir"),
                "trial_valid_for_freeze_candidate": validity.get("run_valid_for_freeze_candidate"),
                "freeze_status": freeze.get("status"),
                "c2_alignment_status": alignment.get("status"),
                "hardware_confounds": list(_sequence(validity.get("hardware_confounds", []))),
            }
        )
        if validity.get("run_valid_for_freeze_candidate"):
            freeze_valid_reports.append(report)

        for row in _sequence_of_mappings(report.get("d2_coordination_trials", [])):
            if not row.get("trial_valid_for_analysis"):
                continue
            key = (str(row.get("task")), str(row.get("candidate_id")))
            task_candidate_trials[key].append(dict(row))

        for ranking in _sequence_of_mappings(_mapping(report.get("d3_summary", {})).get("task_rankings", [])):
            winner = ranking.get("winner_candidate_id")
            task = str(ranking.get("task"))
            if winner:
                winner_counts[task][str(winner)] += 1

    aggregated_stats = _coordination_stats(
        [dict(row) for rows in task_candidate_trials.values() for row in rows]
    )
    aggregated_task_rankings = _task_rankings_from_stats(
        aggregated_stats,
        reference_candidate_id=_infer_reference_candidate(normalized_reports[0]),
    )
    freeze_ready = _cross_run_freeze_ready(
        reports=freeze_valid_reports,
        min_clean_runs_for_freeze=min_clean_runs_for_freeze,
    )

    return {
        "schema": "matrix_d_real_cross_run_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_count": len(normalized_reports),
        "min_clean_runs_for_freeze": int(min_clean_runs_for_freeze),
        "run_rows": run_rows,
        "aggregated_coordination_stats": aggregated_stats,
        "aggregated_task_rankings": aggregated_task_rankings,
        "winner_counts_by_task": {
            task: dict(counter)
            for task, counter in sorted(winner_counts.items())
        },
        "freeze_policy": freeze_ready,
        "paper_ready_metrics": {
            "aggregated_coordination_rows": _paper_coordination_rows(
                aggregated_stats,
                reference_candidate_id=_infer_reference_candidate(normalized_reports[0]),
                trial_validity_label="analysis_aggregate",
            ),
            "aggregated_ranking_rows": _paper_ranking_rows(aggregated_task_rankings),
        },
        "notes": [
            "Cross-run aggregation is exploratory unless freeze-valid clean runs satisfy the freeze policy.",
            "Use run-level reports to mark pilot or confounded runs before aggregation decisions.",
        ],
    }


def _candidate_rows(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    return _sequence_of_mappings(manifest.get("candidate_validations", []))


def _execution_summary(execution: Mapping[str, object]) -> dict[str, Any]:
    steps = _sequence_of_mappings(execution.get("steps", []))
    statuses = Counter(str(step.get("status", "unknown")) for step in steps)
    by_candidate: dict[str, Counter[str]] = defaultdict(Counter)
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    failed_steps: list[dict[str, Any]] = []
    for step in steps:
        candidate_id = str(step.get("candidate_id", "unknown"))
        stage = str(step.get("stage", "unknown"))
        status = str(step.get("status", "unknown"))
        by_candidate[candidate_id][status] += 1
        by_stage[stage][status] += 1
        if status == "failed":
            failed_steps.append(
                {
                    "candidate_id": candidate_id,
                    "step_id": step.get("step_id"),
                    "stage": stage,
                }
            )

    return {
        "step_count": len(steps),
        "status_counts": dict(sorted(statuses.items())),
        "failed_steps": failed_steps,
        "by_candidate": {
            candidate_id: dict(sorted(counter.items()))
            for candidate_id, counter in sorted(by_candidate.items())
        },
        "by_stage": {
            stage: dict(sorted(counter.items()))
            for stage, counter in sorted(by_stage.items())
        },
    }


def _collect_d1_summary(
    run_path: Path,
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    candidate_ids = [
        str(row.get("candidate_id"))
        for row in candidate_rows
        if row.get("candidate_id") is not None
    ]
    coupling_rows: list[dict[str, Any]] = []
    lemma_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_dir = run_path / candidate_id
        for path in sorted(candidate_dir.glob("coupling_*/*.json")):
            payload = _load_json(path)
            coupling_rows.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(path),
                    "config_name": payload.get("config_name"),
                    "perturbed_joint_idx": payload.get("perturbed_joint_idx"),
                    "n_samples": payload.get("n_samples"),
                    "timestamp_utc": payload.get("timestamp_utc"),
                }
            )
        for path in sorted(candidate_dir.glob("lemma3/*.json")):
            payload = _load_json(path)
            lemma_rows.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(path),
                    "trial_count": len(_sequence(payload.get("trials", []))),
                    "timestamp_utc": payload.get("timestamp_utc"),
                }
            )
    return {
        "coupling_runs": coupling_rows,
        "lemma3_runs": lemma_rows,
    }


def _collect_coordination_trials(
    run_path: Path,
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    annotation: Mapping[str, object],
) -> list[dict[str, Any]]:
    route_by_candidate = {
        str(row.get("candidate_id")): str(row.get("route_status", "unknown"))
        for row in candidate_rows
        if row.get("candidate_id") is not None
    }
    trial_overrides = _trial_override_map(annotation)
    trials: list[dict[str, Any]] = []
    for candidate_id, route_status in sorted(route_by_candidate.items()):
        candidate_dir = run_path / candidate_id
        for path in sorted(candidate_dir.glob("coordination_*/*.json")):
            payload = _load_json(path)
            task = str(payload.get("task") or path.parent.name.removeprefix("coordination_"))
            rep = int(payload.get("rep", 0))
            override = trial_overrides.get((candidate_id, task, rep), {})
            grasp_mode = str(
                override.get("grasp_mode")
                or annotation.get("grasp_mode")
                or "unknown"
            )
            hardware_confounds = list(
                _sequence(override.get("hardware_confounds", annotation.get("hardware_confounds", [])))
            )
            trial_notes = list(_sequence(override.get("notes", annotation.get("operator_notes", []))))
            metric_present = all(
                payload.get(key) is not None
                for key in ("rmse_total", "rmse_right", "rmse_left", "n_samples")
            )
            analysis_valid = bool(metric_present) and int(payload.get("n_samples", 0)) > 0
            freeze_valid = (
                analysis_valid
                and grasp_mode == "true_grasp"
                and len(hardware_confounds) == 0
                and not bool(override.get("force_invalid_for_freeze", False))
            )
            reason_codes = []
            if not analysis_valid:
                reason_codes.append("missing_or_invalid_metrics")
            if grasp_mode != "true_grasp":
                reason_codes.append(f"grasp_mode={grasp_mode}")
            if hardware_confounds:
                reason_codes.append("hardware_confounds_present")
            if bool(override.get("force_invalid_for_freeze", False)):
                reason_codes.append("force_invalid_for_freeze")

            trials.append(
                {
                    "candidate_id": candidate_id,
                    "route_status": route_status,
                    "task": task,
                    "controller": payload.get("controller"),
                    "config": payload.get("config"),
                    "rep": rep,
                    "rmse_right": _as_float(payload.get("rmse_right")),
                    "rmse_left": _as_float(payload.get("rmse_left")),
                    "rmse_total": _as_float(payload.get("rmse_total")),
                    "s_rho_l": _as_float(payload.get("s_rho_l")),
                    "j_cross_max": _as_float(payload.get("j_cross_max")),
                    "n_samples": int(payload.get("n_samples", 0)),
                    "timestamp_utc": payload.get("timestamp_utc"),
                    "path": str(path),
                    "grasp_mode": grasp_mode,
                    "hardware_confounds": hardware_confounds,
                    "notes": trial_notes,
                    "trial_valid_for_analysis": analysis_valid,
                    "trial_valid_for_freeze": freeze_valid,
                    "trial_validity_label": (
                        "valid_for_freeze"
                        if freeze_valid
                        else ("valid_for_analysis_only" if analysis_valid else "invalid")
                    ),
                    "trial_validity_reasons": reason_codes,
                }
            )
    return trials


def _coordination_stats(trials: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in trials:
        if not row.get("trial_valid_for_analysis"):
            continue
        key = (str(row.get("candidate_id")), str(row.get("task")))
        grouped[key].append(row)

    stats_rows: list[dict[str, Any]] = []
    for (candidate_id, task), rows in sorted(grouped.items()):
        rmse_total = [_as_float(row.get("rmse_total")) for row in rows]
        rmse_right = [_as_float(row.get("rmse_right")) for row in rows]
        rmse_left = [_as_float(row.get("rmse_left")) for row in rows]
        n_samples = [int(row.get("n_samples", 0)) for row in rows]
        freeze_valid_count = sum(1 for row in rows if row.get("trial_valid_for_freeze"))
        stats_rows.append(
            {
                "candidate_id": candidate_id,
                "task": task,
                "trial_count": len(rows),
                "freeze_valid_trial_count": freeze_valid_count,
                "rmse_total_mean": _mean(rmse_total),
                "rmse_total_std": _std(rmse_total),
                "rmse_total_sem": _sem(rmse_total),
                "rmse_total_median": _median(rmse_total),
                "rmse_total_min": min(rmse_total) if rmse_total else 0.0,
                "rmse_total_max": max(rmse_total) if rmse_total else 0.0,
                "rmse_right_mean": _mean(rmse_right),
                "rmse_left_mean": _mean(rmse_left),
                "mean_abs_left_right_gap": _mean(
                    [abs(r - l) for r, l in zip(rmse_right, rmse_left, strict=False)]
                ),
                "n_samples_mean": _mean([float(value) for value in n_samples]),
                "reps": sorted(int(row.get("rep", 0)) for row in rows),
                "grasp_modes": sorted({str(row.get("grasp_mode")) for row in rows}),
                "hardware_confounds": sorted(
                    {
                        str(item)
                        for row in rows
                        for item in _sequence(row.get("hardware_confounds", []))
                    }
                ),
            }
        )
    return stats_rows


def _task_rankings(
    coordination_stats: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> list[dict[str, Any]]:
    reference_candidate_id = str(manifest.get("reference_candidate_id"))
    return _task_rankings_from_stats(
        coordination_stats,
        reference_candidate_id=reference_candidate_id,
    )


def _task_rankings_from_stats(
    coordination_stats: Sequence[Mapping[str, object]],
    reference_candidate_id: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in coordination_stats:
        grouped[str(row.get("task"))].append(row)

    rankings: list[dict[str, Any]] = []
    for task, rows in sorted(grouped.items()):
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                _as_float(row.get("rmse_total_mean")),
                str(row.get("candidate_id")),
            ),
        )
        reference_mean = None
        if reference_candidate_id:
            for row in sorted_rows:
                if str(row.get("candidate_id")) == reference_candidate_id:
                    reference_mean = _as_float(row.get("rmse_total_mean"))
                    break
        ranking_rows = []
        for rank, row in enumerate(sorted_rows, start=1):
            mean_value = _as_float(row.get("rmse_total_mean"))
            improvement = (
                (reference_mean - mean_value)
                if reference_mean is not None
                else None
            )
            ranking_rows.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "rank": rank,
                    "rmse_total_mean": mean_value,
                    "rmse_total_std": _as_float(row.get("rmse_total_std")),
                    "trial_count": int(row.get("trial_count", 0)),
                    "baseline_relative_improvement": improvement,
                    "baseline_relative_improvement_pct": (
                        (100.0 * improvement / reference_mean)
                        if reference_mean not in (None, 0.0) and improvement is not None
                        else None
                    ),
                }
            )
        rankings.append(
            {
                "task": task,
                "metric": "rmse_total_mean",
                "direction": "lower_is_better",
                "reference_candidate_id": reference_candidate_id,
                "winner_candidate_id": ranking_rows[0]["candidate_id"] if ranking_rows else None,
                "ranking": ranking_rows,
            }
        )
    return rankings


def _pairwise_vs_reference(
    coordination_stats: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> list[dict[str, Any]]:
    reference_candidate_id = str(manifest.get("reference_candidate_id"))
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in coordination_stats:
        grouped[str(row.get("task"))].append(row)

    outputs: list[dict[str, Any]] = []
    for task, rows in sorted(grouped.items()):
        reference_row = next(
            (row for row in rows if str(row.get("candidate_id")) == reference_candidate_id),
            None,
        )
        if reference_row is None:
            continue
        ref_mean = _as_float(reference_row.get("rmse_total_mean"))
        ref_std = _as_float(reference_row.get("rmse_total_std"))
        ref_n = int(reference_row.get("trial_count", 0))
        for row in rows:
            candidate_id = str(row.get("candidate_id"))
            if candidate_id == reference_candidate_id:
                continue
            cand_mean = _as_float(row.get("rmse_total_mean"))
            cand_std = _as_float(row.get("rmse_total_std"))
            cand_n = int(row.get("trial_count", 0))
            delta = cand_mean - ref_mean
            outputs.append(
                {
                    "task": task,
                    "reference_candidate_id": reference_candidate_id,
                    "candidate_id": candidate_id,
                    "delta_rmse_total_mean": delta,
                    "improvement_vs_reference": ref_mean - cand_mean,
                    "improvement_vs_reference_pct": (
                        100.0 * (ref_mean - cand_mean) / ref_mean if ref_mean else None
                    ),
                    "cohen_d": _cohen_d(
                        mean_a=cand_mean,
                        std_a=cand_std,
                        n_a=cand_n,
                        mean_b=ref_mean,
                        std_b=ref_std,
                        n_b=ref_n,
                    ),
                    "welch_t_like": _welch_t_like(
                        mean_a=cand_mean,
                        std_a=cand_std,
                        n_a=cand_n,
                        mean_b=ref_mean,
                        std_b=ref_std,
                        n_b=ref_n,
                    ),
                    "candidate_trial_count": cand_n,
                    "reference_trial_count": ref_n,
                }
            )
    return outputs


def _c2_alignment(
    task_rankings: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> dict[str, Any]:
    primary_ids = _route_candidate_ids(manifest, "primary_validate")
    secondary_ids = _route_candidate_ids(manifest, "secondary_validate")
    reference_candidate_id = str(manifest.get("reference_candidate_id"))

    tested_tasks: list[str] = []
    primary_first_tasks: list[str] = []
    primary_beats_reference_tasks: list[str] = []
    secondary_present_tasks: list[str] = []
    order_support_tasks: list[str] = []

    for ranking in task_rankings:
        task = str(ranking.get("task"))
        rows = _sequence_of_mappings(ranking.get("ranking", []))
        rank_by_id = {str(row.get("candidate_id")): int(row.get("rank", 0)) for row in rows}
        if not primary_ids:
            continue
        primary_id = primary_ids[0]
        if primary_id not in rank_by_id or reference_candidate_id not in rank_by_id:
            continue
        tested_tasks.append(task)
        if rank_by_id[primary_id] == 1:
            primary_first_tasks.append(task)
        if rank_by_id[primary_id] < rank_by_id[reference_candidate_id]:
            primary_beats_reference_tasks.append(task)
        if secondary_ids and secondary_ids[0] in rank_by_id:
            secondary_present_tasks.append(task)
            if (
                rank_by_id[primary_id] < rank_by_id[secondary_ids[0]]
                and rank_by_id[secondary_ids[0]] < rank_by_id[reference_candidate_id]
            ):
                order_support_tasks.append(task)

    if not tested_tasks:
        status = "not_enough_downstream_tasks"
    elif tested_tasks == primary_first_tasks == order_support_tasks:
        status = "supports_c2_routing"
    elif tested_tasks == primary_beats_reference_tasks:
        status = "supports_primary_over_reference_only"
    else:
        status = "mixed_or_not_supported"

    return {
        "status": status,
        "tested_tasks": tested_tasks,
        "primary_first_tasks": primary_first_tasks,
        "primary_beats_reference_tasks": primary_beats_reference_tasks,
        "secondary_present_tasks": secondary_present_tasks,
        "order_support_tasks": order_support_tasks,
        "expected_primary_candidate_id": primary_ids[0] if primary_ids else None,
        "expected_secondary_candidate_id": secondary_ids[0] if secondary_ids else None,
        "reference_candidate_id": reference_candidate_id,
    }


def _trial_validity_gate(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    coordination_trials: Sequence[Mapping[str, object]],
    execution_summary: Mapping[str, object],
    expected_tasks: Sequence[object],
    expected_reps: int,
    annotation: Mapping[str, object],
) -> dict[str, Any]:
    scheduled_candidates = [
        str(row.get("candidate_id"))
        for row in candidate_rows
        if bool(row.get("scheduled_for_execution"))
    ]
    trial_map: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in coordination_trials:
        key = (str(row.get("candidate_id")), str(row.get("task")))
        trial_map[key].append(row)

    coverage_rows = []
    reasons: list[str] = []
    for candidate_id in scheduled_candidates:
        for task_obj in expected_tasks:
            task = str(task_obj)
            rows = trial_map.get((candidate_id, task), [])
            reps_observed = len(rows)
            analysis_valid_count = sum(1 for row in rows if row.get("trial_valid_for_analysis"))
            freeze_valid_count = sum(1 for row in rows if row.get("trial_valid_for_freeze"))
            coverage_ok = reps_observed >= expected_reps if expected_reps > 0 else reps_observed > 0
            freeze_ok = coverage_ok and (
                freeze_valid_count >= expected_reps if expected_reps > 0 else freeze_valid_count > 0
            )
            if not coverage_ok:
                reasons.append(f"missing_trials:{candidate_id}:{task}")
            if coverage_ok and not freeze_ok:
                reasons.append(f"freeze_invalid_trials:{candidate_id}:{task}")
            coverage_rows.append(
                {
                    "candidate_id": candidate_id,
                    "task": task,
                    "expected_reps": expected_reps,
                    "reps_observed": reps_observed,
                    "analysis_valid_count": analysis_valid_count,
                    "freeze_valid_count": freeze_valid_count,
                    "coverage_ok": coverage_ok,
                    "freeze_ok": freeze_ok,
                }
            )

    failed_steps = list(_sequence(_mapping(execution_summary).get("failed_steps", [])))
    if failed_steps:
        reasons.append("failed_execution_steps")
    hardware_confounds = list(_sequence(annotation.get("hardware_confounds", [])))
    if hardware_confounds:
        reasons.append("hardware_confounds_present")
    grasp_mode = str(annotation.get("grasp_mode", "unknown"))
    if grasp_mode != "true_grasp":
        reasons.append(f"grasp_mode={grasp_mode}")

    unique_reasons = sorted(set(reasons))
    run_valid_for_analysis = len(failed_steps) == 0 and len(coordination_trials) > 0
    run_valid_for_freeze_candidate = (
        run_valid_for_analysis
        and len(unique_reasons) == 0
        and all(bool(row["freeze_ok"]) for row in coverage_rows)
    )
    return {
        "grasp_mode": grasp_mode,
        "hardware_confounds": hardware_confounds,
        "operator_notes": list(_sequence(annotation.get("operator_notes", []))),
        "scheduled_candidate_ids": scheduled_candidates,
        "coverage_rows": coverage_rows,
        "failed_step_count": len(failed_steps),
        "trial_count": len(coordination_trials),
        "trial_valid_for_analysis_count": sum(
            1 for row in coordination_trials if row.get("trial_valid_for_analysis")
        ),
        "trial_valid_for_freeze_count": sum(
            1 for row in coordination_trials if row.get("trial_valid_for_freeze")
        ),
        "run_valid_for_analysis": run_valid_for_analysis,
        "run_valid_for_freeze_candidate": run_valid_for_freeze_candidate,
        "rejection_reasons": unique_reasons,
    }


def _freeze_policy(
    *,
    trial_validity_gate: Mapping[str, object],
    c2_alignment: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, Any]:
    if not trial_validity_gate.get("run_valid_for_analysis"):
        status = "not_analysis_valid"
    elif not trial_validity_gate.get("run_valid_for_freeze_candidate"):
        status = "pilot_or_confounded_not_freeze_ready"
    elif c2_alignment.get("status") == "supports_c2_routing":
        status = "clean_supports_c2_routing_provisional_freeze_candidate"
    elif c2_alignment.get("status") == "supports_primary_over_reference_only":
        status = "clean_but_only_partial_c2_support"
    else:
        status = "clean_but_mixed_downstream_alignment"

    return {
        "status": status,
        "reference_candidate_id": manifest.get("reference_candidate_id"),
        "requires_confirmatory_clean_rerun": True,
        "run_valid_for_freeze_candidate": trial_validity_gate.get(
            "run_valid_for_freeze_candidate"
        ),
        "c2_alignment_status": c2_alignment.get("status"),
        "reasons": list(_sequence(trial_validity_gate.get("rejection_reasons", []))),
    }


def _paper_ready_metrics(
    *,
    coordination_stats: Sequence[Mapping[str, object]],
    task_rankings: Sequence[Mapping[str, object]],
    pairwise_vs_reference: Sequence[Mapping[str, object]],
    trial_validity_gate: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, Any]:
    reference_candidate_id = str(manifest.get("reference_candidate_id"))
    validity_label = (
        "freeze_candidate"
        if trial_validity_gate.get("run_valid_for_freeze_candidate")
        else "analysis_only"
    )
    return {
        "coordination_table_rows": _paper_coordination_rows(
            coordination_stats,
            reference_candidate_id=reference_candidate_id,
            trial_validity_label=validity_label,
        ),
        "ranking_table_rows": _paper_ranking_rows(task_rankings),
        "baseline_delta_rows": [
            {
                "task": row.get("task"),
                "candidate_id": row.get("candidate_id"),
                "reference_candidate_id": row.get("reference_candidate_id"),
                "delta_rmse_total_mean": row.get("delta_rmse_total_mean"),
                "improvement_vs_reference": row.get("improvement_vs_reference"),
                "improvement_vs_reference_pct": row.get("improvement_vs_reference_pct"),
                "cohen_d": row.get("cohen_d"),
            }
            for row in pairwise_vs_reference
        ],
    }


def _paper_coordination_rows(
    coordination_stats: Sequence[Mapping[str, object]],
    *,
    reference_candidate_id: str | None,
    trial_validity_label: str,
) -> list[dict[str, Any]]:
    reference_map = {
        str(row.get("task")): _as_float(row.get("rmse_total_mean"))
        for row in coordination_stats
        if reference_candidate_id is not None
        and str(row.get("candidate_id")) == reference_candidate_id
    }
    rows = []
    for row in coordination_stats:
        task = str(row.get("task"))
        candidate_id = str(row.get("candidate_id"))
        reference_mean = reference_map.get(task)
        mean_value = _as_float(row.get("rmse_total_mean"))
        improvement = reference_mean - mean_value if reference_mean is not None else None
        rows.append(
            {
                "task": task,
                "candidate_id": candidate_id,
                "rmse_total_mean": mean_value,
                "rmse_total_std": _as_float(row.get("rmse_total_std")),
                "rmse_right_mean": _as_float(row.get("rmse_right_mean")),
                "rmse_left_mean": _as_float(row.get("rmse_left_mean")),
                "trial_count": int(row.get("trial_count", 0)),
                "freeze_valid_trial_count": int(row.get("freeze_valid_trial_count", 0)),
                "baseline_relative_improvement": improvement,
                "baseline_relative_improvement_pct": (
                    (100.0 * improvement / reference_mean)
                    if reference_mean not in (None, 0.0) and improvement is not None
                    else None
                ),
                "trial_validity_label": trial_validity_label,
            }
        )
    return rows


def _paper_ranking_rows(task_rankings: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    rows = []
    for ranking in task_rankings:
        task = str(ranking.get("task"))
        for row in _sequence_of_mappings(ranking.get("ranking", [])):
            rows.append(
                {
                    "task": task,
                    "candidate_id": row.get("candidate_id"),
                    "rank": row.get("rank"),
                    "rmse_total_mean": row.get("rmse_total_mean"),
                    "baseline_relative_improvement": row.get("baseline_relative_improvement"),
                    "baseline_relative_improvement_pct": row.get(
                        "baseline_relative_improvement_pct"
                    ),
                }
            )
    return rows


def _cross_run_freeze_ready(
    *,
    reports: Sequence[Mapping[str, object]],
    min_clean_runs_for_freeze: int,
) -> dict[str, Any]:
    clean_reports = [
        report
        for report in reports
        if _mapping(_mapping(report.get("statistical_report", {})).get("c2_alignment", {})).get(
            "status"
        )
        == "supports_c2_routing"
    ]
    if len(clean_reports) >= min_clean_runs_for_freeze:
        status = "freeze_ready"
    elif clean_reports:
        status = "promising_but_needs_more_clean_runs"
    else:
        status = "not_freeze_ready"
    return {
        "status": status,
        "clean_run_count": len(clean_reports),
        "min_clean_runs_for_freeze": min_clean_runs_for_freeze,
        "clean_run_ids": [report.get("run_id") for report in clean_reports],
    }


def _candidate_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.json"))


def _normalize_annotation(annotation: Mapping[str, object] | None) -> dict[str, Any]:
    base = {
        "grasp_mode": "unknown",
        "hardware_confounds": [],
        "operator_notes": [],
        "trial_overrides": [],
    }
    if annotation is None:
        return base
    merged = {**base, **dict(annotation)}
    merged["hardware_confounds"] = [
        str(value) for value in _sequence(merged.get("hardware_confounds", []))
    ]
    merged["operator_notes"] = [
        str(value) for value in _sequence(merged.get("operator_notes", []))
    ]
    merged["trial_overrides"] = [
        dict(value) for value in _sequence_of_mappings(merged.get("trial_overrides", []))
    ]
    return merged


def _trial_override_map(
    annotation: Mapping[str, object],
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    overrides = {}
    for row in _sequence_of_mappings(annotation.get("trial_overrides", [])):
        candidate_id = row.get("candidate_id")
        task = row.get("task")
        rep = row.get("rep")
        if candidate_id is None or task is None or rep is None:
            continue
        overrides[(str(candidate_id), str(task), int(rep))] = row
    return overrides


def _infer_reference_candidate(report: Mapping[str, object]) -> str | None:
    reference = report.get("reference_candidate_id")
    return str(reference) if reference is not None else None


def _route_candidate_ids(
    manifest: Mapping[str, object],
    route_status: str,
) -> list[str]:
    rows = _sequence_of_mappings(manifest.get("candidate_validations", []))
    return [
        str(row.get("candidate_id"))
        for row in rows
        if str(row.get("route_status")) == route_status and row.get("candidate_id") is not None
    ]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sequence(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(value)


def _sequence_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(median(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def _sem(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(_std(values) / math.sqrt(len(values)))


def _cohen_d(
    *,
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float | None:
    if n_a <= 1 or n_b <= 1:
        return None
    pooled_num = ((n_a - 1) * (std_a**2)) + ((n_b - 1) * (std_b**2))
    pooled_den = n_a + n_b - 2
    if pooled_den <= 0:
        return None
    pooled_std = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled_std == 0.0:
        return None
    return float((mean_a - mean_b) / pooled_std)


def _welch_t_like(
    *,
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float | None:
    if n_a <= 0 or n_b <= 0:
        return None
    denom = math.sqrt(((std_a**2) / max(n_a, 1)) + ((std_b**2) / max(n_b, 1)))
    if denom == 0.0:
        return None
    return float((mean_a - mean_b) / denom)
