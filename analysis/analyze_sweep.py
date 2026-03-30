#!/usr/bin/env python3
"""Analyze elbow configuration sweep results.

Tracks how coupling landscape changes as q4 (elbow) sweeps through a range.
Compares measured torque coupling with CRBA-predicted J_ij at each step.

Usage:
    python analysis/analyze_sweep.py results/openarm_elbow_sweep.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.experiment.signal_processing import extract_coupling_amplitudes


def analyze(fpath: Path) -> dict:
    with open(fpath) as f:
        data = json.load(f)

    perturb_idx = data["perturb_joint_idx"]
    n_joints = 7
    steps = []

    for step in data["steps"]:
        q4_deg = step["q4_deg"]
        torques = np.array(step["torques_Nm"])
        ts = np.array(step["timestamps_s"])
        J_pred = np.array(step["theoretical"]["J_matrix"])

        # Extract coupling amplitudes from torque data
        freq = 0.5  # default perturbation frequency
        amps = extract_coupling_amplitudes(torques, ts, freq, ramp_s=1.5)

        # Normalize by self-response
        self_amp = amps[perturb_idx]
        if self_amp > 1e-10:
            c_emp = amps / self_amp
        else:
            c_emp = np.zeros(n_joints)

        # Predicted couplings from this perturbed joint
        j_row = np.abs(J_pred[perturb_idx])

        # Correlation between predicted and measured (excluding self)
        mask = np.ones(n_joints, dtype=bool)
        mask[perturb_idx] = False
        c_other = c_emp[mask]
        j_other = j_row[mask]

        from scipy.stats import pearsonr
        if np.std(c_other) > 1e-10 and np.std(j_other) > 1e-10:
            r, p = pearsonr(j_other, c_other)
        else:
            r, p = 0.0, 1.0

        steps.append({
            "q4_deg": float(q4_deg),
            "amplitudes_Nm": amps.tolist(),
            "c_emp_normalized": c_emp.tolist(),
            "j_pred_row": j_row.tolist(),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "max_coupling_joint": int(np.argmax(c_emp[mask])),
            "max_coupling_value": float(np.max(c_emp[mask])),
            "max_predicted_joint": int(np.argmax(j_other)),
            "max_predicted_value": float(np.max(j_other)),
        })

    # Overall tracking: does the strongest coupling joint shift as predicted?
    pred_dominant = [s["max_predicted_joint"] for s in steps]
    meas_dominant = [s["max_coupling_joint"] for s in steps]
    match_rate = sum(1 for p, m in zip(pred_dominant, meas_dominant) if p == m) / len(steps)

    rs = [s["pearson_r"] for s in steps]

    result = {
        "experiment": "elbow_sweep_analysis",
        "source_file": str(fpath),
        "perturb_joint_idx": perturb_idx,
        "sweep_joint_idx": data.get("sweep_joint_idx", 3),
        "n_steps": len(steps),
        "mean_pearson_r": float(np.mean(rs)),
        "min_pearson_r": float(np.min(rs)),
        "max_pearson_r": float(np.max(rs)),
        "dominant_joint_match_rate": float(match_rate),
        "steps": steps,
    }

    return result


def main():
    if len(sys.argv) < 2:
        fpath = Path("results/openarm_elbow_sweep.json")
    else:
        fpath = Path(sys.argv[1])

    if not fpath.exists():
        print(f"File not found: {fpath}")
        sys.exit(1)

    print(f"Analyzing: {fpath}")
    result = analyze(fpath)

    print(f"\n=== Elbow Sweep Analysis ===")
    print(f"  Steps: {result['n_steps']}")
    print(f"  Perturbed joint: {result['perturb_joint_idx']}")
    print(f"  Mean Pearson r: {result['mean_pearson_r']:.3f}")
    print(f"  Min/Max r: {result['min_pearson_r']:.3f} / {result['max_pearson_r']:.3f}")
    print(f"  Dominant joint match rate: {result['dominant_joint_match_rate']:.1%}")

    print(f"\n  Per-step:")
    for s in result["steps"]:
        print(f"    q4={s['q4_deg']:+6.1f} deg: r={s['pearson_r']:.3f}, "
              f"strongest={s['max_coupling_joint']} (pred={s['max_predicted_joint']})")

    out = Path("results/sweep_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
