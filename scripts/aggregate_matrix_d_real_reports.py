#!/usr/bin/env python3
"""Aggregate multiple Matrix D real-robot report JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsreal.experiment.matrix_d_real_report import build_matrix_d_real_cross_run_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Matrix D real reports")
    parser.add_argument(
        "--report-input",
        action="append",
        required=True,
        help="Path to a matrix_d_real_report_v1 JSON; repeatable",
    )
    parser.add_argument("--min-clean-runs-for-freeze", type=int, default=2)
    parser.add_argument("--output", required=True, help="Output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.report_input
    ]
    payload = build_matrix_d_real_cross_run_report(
        reports,
        min_clean_runs_for_freeze=args.min_clean_runs_for_freeze,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved Matrix D cross-run report to {output_path}")
    print(
        "  status={status}  clean_runs={count}/{needed}".format(
            status=payload["freeze_policy"]["status"],
            count=payload["freeze_policy"]["clean_run_count"],
            needed=payload["freeze_policy"]["min_clean_runs_for_freeze"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
