from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_module():
    script_path = _repo_root() / "scripts" / "identify_control_hardware_response.py"
    spec = importlib.util.spec_from_file_location("identify_control_hardware_response", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_response_matrix_from_column_trials_recovers_simulated_template() -> None:
    module = _load_module()
    base = np.zeros(module.N_TOTAL, dtype=np.float64)
    columns = [2, 10]
    trials = [
        module._simulate_column_trial(
            candidate_id="s_adaptive_entropy",
            column_index=column,
            base_target_deg=base,
            amplitude_deg=1.0,
            frequency_hz=0.5,
            duration_s=1.0,
            dt=0.05,
            ramp_s=0.2,
        )
        for column in columns
    ]

    response, metrics = module._response_matrix_from_column_trials(
        trials,
        n_total=module.N_TOTAL,
        base_target_deg=base,
        settle_fraction=0.2,
    )

    expected = module._simulate_response_template("s_adaptive_entropy", module.N_TOTAL)
    for column in columns:
        assert np.allclose(response[:, column], expected[:, column], atol=1.0e-6)
    assert [row["column_index"] for row in metrics] == columns


def test_script_dry_run_writes_matrix_b_compatible_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "hardware_response.json"
    script_path = _repo_root() / "scripts" / "identify_control_hardware_response.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dry-run",
            "--context",
            "bar_only:bar_b",
            "--candidate-id",
            "decoupled_ref",
            "--columns",
            "2",
            "10",
            "--duration",
            "0.2",
            "--dt",
            "0.05",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=str(_repo_root()),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "matrix_b_hardware_response_models_v1"
    assert payload["response_source"] == "hardware_identified_v1"
    assert "bar_only:bar_b" in payload
    context_payload = payload["bar_only:bar_b"]
    assert "decoupled_ref" in context_payload
    candidate_payload = context_payload["decoupled_ref"]
    assert candidate_payload["response_source"] == "hardware_identified_v1"
    assert candidate_payload["response_source_details"]["identified_columns"] == [2, 10]
    assert len(candidate_payload["response_matrix"]) == 14
    assert len(candidate_payload["response_matrix"][0]) == 14
