#!/usr/bin/env python3
"""Analyze Piper position-coupling experiment results.

Since Piper provides only position readouts (no torque), we extract
coupling by measuring position oscillation amplitude at non-perturbed
joints, which reflects dynamic coupling through M_ij.

Usage:
    python analysis/analyze_piper.py results/piper_coupling_home_j*.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.experiment.signal_processing import (
    extract_coupling_amplitudes,
    build_empirical_coupling_matrix,
    compare_with_theory,
)


def extract_position_deviations(positions, commanded, timestamps,
                                 frequency_hz=0.5, ramp_s=2.0):
    """Extract position deviation amplitudes at perturbation frequency.

    Parameters
    ----------
    positions : (N, n_joints) actual positions (deg)
    commanded : (N, n_joints) commanded positions (deg)
    timestamps : (N,) timestamps (s)

    Returns
    -------
    amplitudes : (n_joints,) position deviation amplitudes at freq (deg)
    """
    deviation = positions - commanded
    return extract_coupling_amplitudes(deviation, timestamps, frequency_hz, ramp_s)


def analyze_config(json_files):
    """Analyze all trials from one Piper configuration."""
    n_joints = 6
    all_amplitudes = {}

    for fpath in sorted(json_files):
        with open(fpath) as f:
            data = json.load(f)

        j_idx = data["perturbed_joint_idx"]
        positions = np.array(data["positions_deg"])
        commanded = np.array(data["commanded_deg"])
        timestamps = np.array(data["timestamps_s"])
        freq = data["perturbation"]["frequency_hz"]
        ramp = data["perturbation"]["ramp_s"]

        # Method 1: raw position oscillation (captures coupling-induced motion)
        amps_raw = extract_coupling_amplitudes(positions, timestamps, freq, ramp)

        # Method 2: position deviation from commanded (purer coupling signal)
        amps_dev = extract_position_deviations(
            positions, commanded, timestamps, freq, ramp
        )

        # Use raw position amplitudes as primary (simpler, more robust)
        all_amplitudes[j_idx] = amps_raw

        print(f"  {fpath.name}: perturb j{j_idx}, "
              f"self_amp={amps_raw[j_idx]:.4f}, "
              f"max_other_raw={np.max(np.delete(amps_raw, j_idx)):.4f}, "
              f"max_other_dev={np.max(np.delete(amps_dev, j_idx)):.4f}")

    if not all_amplitudes:
        print("  No data found!")
        return None

    C_emp = build_empirical_coupling_matrix(all_amplitudes, n_joints)

    sample = json.loads(open(json_files[0]).read())
    J_pred = np.array(sample["theoretical"]["J_matrix"])

    d_kin = np.zeros((n_joints, n_joints))
    for i in range(n_joints):
        for j in range(n_joints):
            d_kin[i, j] = abs(i - j)

    result = compare_with_theory(C_emp, J_pred, d_kin)

    config_name = sample["config_name"]
    print(f"\n  === {config_name} ===")
    print(f"  Pearson r(|J|, C_emp) = {result['pearson_r']:.3f}")
    print(f"  Spearman rho          = {result.get('spearman_rho', 0):.3f}")
    print(f"  Kinematic r(1/d, C)   = {result.get('kinematic_pearson_r', 0):.3f}")

    print(f"\n  Top-5 predicted couplings:")
    pairs = []
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            pairs.append((abs(J_pred[i, j]), C_emp[i, j], i, j))
    pairs.sort(reverse=True)
    for J_val, C_val, i, j in pairs[:5]:
        print(f"    ({i},{j}): |J|={J_val:.4f}, C_emp={C_val:.4f}")

    return {
        "config_name": config_name,
        "robot": "Piper_6DOF",
        "data_type": "position_coupling",
        "C_emp": C_emp.tolist(),
        "J_pred": J_pred.tolist(),
        **result,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_piper.py results/piper_coupling_<config>_j*.json")
        sys.exit(1)

    files = [Path(f) for f in sys.argv[1:]]

    from collections import defaultdict
    by_config = defaultdict(list)
    for f in files:
        parts = f.stem.split("_")
        config_idx = parts.index("coupling") + 1
        config_name = parts[config_idx]
        by_config[config_name].append(f)

    all_results = []
    for config_name, config_files in sorted(by_config.items()):
        print(f"\nAnalyzing config: {config_name} ({len(config_files)} files)")
        result = analyze_config(config_files)
        if result:
            all_results.append(result)

    out = Path("results/piper_coupling_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved analysis to {out}")


if __name__ == "__main__":
    main()
