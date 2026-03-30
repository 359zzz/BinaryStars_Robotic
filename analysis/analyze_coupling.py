#!/usr/bin/env python3
"""Analyze coupling experiment results: extract C_emp, compare with J_ij.

Usage:
    python analysis/analyze_coupling.py results/openarm_coupling_home_j*.json
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


def analyze_config(json_files: list[Path]):
    """Analyze all trials from one configuration."""
    n_joints = 7
    all_amplitudes = {}

    for fpath in sorted(json_files):
        with open(fpath) as f:
            data = json.load(f)

        j_idx = data["perturbed_joint_idx"]
        torques = np.array(data["torques_Nm"])
        timestamps = np.array(data["timestamps_s"])
        freq = data["perturbation"]["frequency_hz"]
        ramp = data["perturbation"]["ramp_s"]

        amps = extract_coupling_amplitudes(torques, timestamps, freq, ramp)
        all_amplitudes[j_idx] = amps

        print(f"  {fpath.name}: perturb j{j_idx}, "
              f"self_amp={amps[j_idx]:.4f}, "
              f"max_other={np.max(np.delete(amps, j_idx)):.4f}")

    if not all_amplitudes:
        print("  No data found!")
        return None

    # Build empirical coupling
    C_emp = build_empirical_coupling_matrix(all_amplitudes, n_joints)

    # Get theoretical J
    sample = json.loads(open(json_files[0]).read())
    J_pred = np.array(sample["theoretical"]["J_matrix"])

    # Kinematic distances (serial chain)
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

    # Show top predicted vs measured
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
        "C_emp": C_emp.tolist(),
        "J_pred": J_pred.tolist(),
        **result,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_coupling.py results/openarm_coupling_<config>_j*.json")
        sys.exit(1)

    files = [Path(f) for f in sys.argv[1:]]
    if not files:
        print("No files found")
        sys.exit(1)

    # Group by config
    from collections import defaultdict
    by_config = defaultdict(list)
    for f in files:
        # Extract config name from filename like openarm_coupling_home_j0.json
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

    # Save analysis
    out = Path("results/coupling_analysis.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved analysis to {out}")


if __name__ == "__main__":
    main()
