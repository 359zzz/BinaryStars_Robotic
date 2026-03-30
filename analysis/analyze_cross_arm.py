#!/usr/bin/env python3
"""Analyze dual-arm Lemma 3 results: cross-arm vs within-arm torque coupling.

Usage:
    python analysis/analyze_cross_arm.py results/dual_arm_lemma3.json
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

    own_amps_all = []
    other_amps_all = []
    trial_summaries = []

    for trial in data["trials"]:
        own_tau = np.array(trial["own_arm_torques_Nm"])
        other_tau = np.array(trial["other_arm_torques_Nm"])
        ts = np.array(trial["timestamps_s"])
        freq = trial["perturbation"]["frequency_hz"]

        own_a = extract_coupling_amplitudes(own_tau, ts, freq, ramp_s=2.0)
        other_a = extract_coupling_amplitudes(other_tau, ts, freq, ramp_s=2.0)

        own_amps_all.extend(own_a.tolist())
        other_amps_all.extend(other_a.tolist())

        trial_summaries.append({
            "perturb_arm": trial["perturb_arm"],
            "perturb_joint": trial["perturb_joint_name"],
            "own_arm_mean_amp": float(np.mean(own_a)),
            "own_arm_max_amp": float(np.max(own_a)),
            "other_arm_mean_amp": float(np.mean(other_a)),
            "other_arm_max_amp": float(np.max(other_a)),
            "ratio_mean": float(np.mean(own_a) / max(np.mean(other_a), 1e-10)),
        })

    own_arr = np.array(own_amps_all)
    other_arr = np.array(other_amps_all)

    result = {
        "experiment": "cross_arm_lemma3_analysis",
        "source_file": str(fpath),
        "n_trials": len(data["trials"]),
        "within_arm": {
            "mean_amp_Nm": float(np.mean(own_arr)),
            "std_amp_Nm": float(np.std(own_arr)),
            "max_amp_Nm": float(np.max(own_arr)),
            "median_amp_Nm": float(np.median(own_arr)),
        },
        "cross_arm": {
            "mean_amp_Nm": float(np.mean(other_arr)),
            "std_amp_Nm": float(np.std(other_arr)),
            "max_amp_Nm": float(np.max(other_arr)),
            "median_amp_Nm": float(np.median(other_arr)),
        },
        "ratio_mean": float(np.mean(own_arr) / max(np.mean(other_arr), 1e-10)),
        "ratio_max": float(np.max(own_arr) / max(np.max(other_arr), 1e-10)),
        "lemma3_verified": bool(np.mean(own_arr) > 10 * np.mean(other_arr)),
        "trial_summaries": trial_summaries,
    }

    return result


def main():
    if len(sys.argv) < 2:
        fpath = Path("results/dual_arm_lemma3.json")
    else:
        fpath = Path(sys.argv[1])

    if not fpath.exists():
        print(f"File not found: {fpath}")
        sys.exit(1)

    print(f"Analyzing: {fpath}")
    result = analyze(fpath)

    print(f"\n=== Lemma 3: Cross-Arm Zero Coupling ===")
    print(f"  Trials: {result['n_trials']}")
    print(f"  Within-arm mean amplitude: {result['within_arm']['mean_amp_Nm']:.4f} Nm")
    print(f"  Cross-arm mean amplitude:  {result['cross_arm']['mean_amp_Nm']:.4f} Nm")
    print(f"  Ratio (within/cross):      {result['ratio_mean']:.1f}x")
    print(f"  Lemma 3 verified (>10x):   {result['lemma3_verified']}")

    print(f"\n  Per-trial breakdown:")
    for ts in result["trial_summaries"]:
        print(f"    {ts['perturb_arm']} {ts['perturb_joint']}: "
              f"own={ts['own_arm_mean_amp']:.4f}, "
              f"other={ts['other_arm_mean_amp']:.4f}, "
              f"ratio={ts['ratio_mean']:.1f}x")

    out = Path("results/cross_arm_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
