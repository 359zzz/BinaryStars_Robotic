#!/usr/bin/env python3
"""Build a single-run Matrix D real-robot report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsreal.experiment.matrix_d_real_report import build_matrix_d_real_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Matrix D real-robot report")
    parser.add_argument("--run-dir", required=True, help="Path to a matrix_d_real_<stamp> run directory")
    parser.add_argument("--annotation-input", default=None, help="Optional JSON annotation file")
    parser.add_argument(
        "--grasp-mode",
        default=None,
        choices=["true_grasp", "wrist_rest", "mixed", "unknown"],
        help="Optional run-level grasp mode override",
    )
    parser.add_argument(
        "--hardware-confound",
        action="append",
        default=[],
        help="Optional run-level hardware confound flag; repeatable",
    )
    parser.add_argument(
        "--operator-note",
        action="append",
        default=[],
        help="Optional run-level operator note; repeatable",
    )
    parser.add_argument("--min-coordination-reps", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional output path")
    return parser.parse_args()


def load_annotation(args: argparse.Namespace) -> dict[str, object]:
    annotation: dict[str, object] = {}
    if args.annotation_input:
        annotation = json.loads(Path(args.annotation_input).read_text(encoding="utf-8"))
    if args.grasp_mode is not None:
        annotation["grasp_mode"] = args.grasp_mode
    if args.hardware_confound:
        annotation["hardware_confounds"] = list(args.hardware_confound)
    if args.operator_note:
        annotation["operator_notes"] = list(args.operator_note)
    return annotation


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    annotation = load_annotation(args)
    payload = build_matrix_d_real_report(
        run_dir,
        annotation=annotation,
        min_coordination_reps=args.min_coordination_reps,
    )

    output_path = (
        Path(args.output)
        if args.output
        else run_dir / "matrix_d_real_report.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved Matrix D real report to {output_path}")
    print(
        "  freeze_status={freeze}  c2_alignment={alignment}  valid_for_freeze={valid}".format(
            freeze=payload["freeze_policy"]["status"],
            alignment=payload["statistical_report"]["c2_alignment"]["status"],
            valid=payload["trial_validity_gate"]["run_valid_for_freeze_candidate"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
