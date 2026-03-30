#!/usr/bin/env python3
"""Diagnostic for OpenArm torque-coupling experiment results.

Run on the IPC directly — only needs numpy + scipy.

Usage:
    python analysis/openarm_diagnostic.py results/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def fft_amplitude(signal_1d, timestamps, target_freq, ramp_s=2.0):
    """Extract amplitude at target_freq via FFT (skip ramp)."""
    mask = timestamps >= ramp_s
    sig_cut = signal_1d[mask]
    ts_cut = timestamps[mask]
    if len(sig_cut) < 20:
        return 0.0
    sig_cut = sig_cut - np.mean(sig_cut)
    dt = np.median(np.diff(ts_cut))
    N = len(sig_cut)
    freqs = np.fft.rfftfreq(N, d=dt)
    mags = 2.0 * np.abs(np.fft.rfft(sig_cut)) / N
    idx = np.argmin(np.abs(freqs - target_freq))
    lo, hi = max(0, idx - 1), min(len(mags) - 1, idx + 1)
    return float(np.max(mags[lo:hi + 1]))


def load_trial(fpath):
    with open(fpath) as f:
        d = json.load(f)
    d["torques_Nm"] = np.array(d["torques_Nm"])
    d["positions_deg"] = np.array(d["positions_deg"])
    d["velocities_deg_s"] = np.array(d["velocities_deg_s"])
    d["timestamps_s"] = np.array(d["timestamps_s"])
    d["_path"] = fpath
    return d


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python openarm_diagnostic.py results/")
        sys.exit(1)

    files = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files.extend(sorted(p.glob("openarm_coupling_*.json")))
        elif p.is_file():
            files.append(p)
    files = sorted(set(files))

    if not files:
        print("No openarm_coupling_*.json files found.")
        sys.exit(1)

    print(f"Found {len(files)} trial files\n")

    by_config = defaultdict(list)
    for f in files:
        d = load_trial(f)
        by_config[d["config_name"]].append(d)

    n_joints = 7

    # ── 1. SENSOR HEALTH CHECK ──────────────────────────────────────────
    print("=" * 85)
    print("1. SENSOR HEALTH CHECK")
    print("=" * 85)
    print("Verifying that torque readings are independent per joint.\n")

    sample = list(by_config.values())[0][0]
    tau = sample["torques_Nm"]  # (N, 7)
    ts = sample["timestamps_s"]
    mask = ts >= 2.0
    tau_ss = tau[mask]

    print(f"  Sample file: {Path(sample['_path']).name}")
    print(f"  Samples: {len(ts)}, post-ramp: {mask.sum()}")
    print(f"\n  {'joint':<8s} │ {'mean τ':>9s} │ {'std τ':>9s} │ {'range τ':>9s} │ unique?")
    print(f"  {'─' * 8}─┼{'─' * 11}┼{'─' * 11}┼{'─' * 11}┼{'─' * 8}")

    means = []
    stds = []
    for j in range(n_joints):
        m = np.mean(tau_ss[:, j])
        s = np.std(tau_ss[:, j])
        r = np.ptp(tau_ss[:, j])
        means.append(m)
        stds.append(s)
        print(f"  joint_{j+1} │ {m:+9.5f} │ {s:9.5f} │ {r:9.5f} │")

    # Check for duplicates (same readings on different joints)
    dup_pairs = []
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            if np.allclose(tau_ss[:, i], tau_ss[:, j], atol=1e-6):
                dup_pairs.append((i, j))

    if dup_pairs:
        print(f"\n  *** WARNING: Identical torque streams detected: {dup_pairs} ***")
        print(f"  *** This suggests CAN ID mapping issues. ***")
    else:
        print(f"\n  OK: All 7 torque channels are independent.")

    # ── 2. PER-TRIAL TORQUE COUPLING ────────────────────────────────────
    print("\n" + "=" * 85)
    print("2. PER-TRIAL TORQUE COUPLING")
    print("=" * 85)

    for config_name in sorted(by_config.keys()):
        trials = by_config[config_name]
        print(f"\n┌── {config_name} ({len(trials)} trials) ──┐")
        print(f"│  {'file':<34s} │ j │ self_τ  │ max_oth │ ratio │ strongest │ pred   │ match │")
        print(f"│{'─' * 34}─┼───┼─────────┼─────────┼───────┼───────────┼────────┼───────│")

        for d in sorted(trials, key=lambda x: x["perturbed_joint_idx"]):
            j_p = d["perturbed_joint_idx"]
            tau = d["torques_Nm"]
            ts = d["timestamps_s"]
            freq = d["perturbation"]["frequency_hz"]
            J = np.array(d["theoretical"]["J_matrix"])

            amps = np.array([fft_amplitude(tau[:, j], ts, freq) for j in range(n_joints)])
            self_amp = amps[j_p]
            other_amps = np.delete(amps, j_p)
            other_idxs = [j for j in range(n_joints) if j != j_p]
            max_other = np.max(other_amps)
            strongest_j = other_idxs[np.argmax(other_amps)]

            # Predicted strongest
            j_row = np.abs(J[j_p]).copy()
            j_row[j_p] = 0
            pred_j = np.argmax(j_row)
            pred_J_val = j_row[pred_j]

            ratio = max_other / max(self_amp, 1e-10)
            match = "YES" if strongest_j == pred_j else "no"

            print(f"│  {Path(d['_path']).name:<34s} │ {j_p} │ {self_amp:7.5f} │ {max_other:7.5f} │"
                  f" {ratio:5.3f} │    j{strongest_j+1}     │ j{pred_j+1} {pred_J_val:.3f} │  {match:>3s}  │")

        print(f"└{'─' * 85}┘")

    # ── 3. COUPLING MATRIX ANALYSIS ─────────────────────────────────────
    print("\n" + "=" * 85)
    print("3. COUPLING MATRIX: C_emp vs |J_pred|")
    print("=" * 85)

    summary = []
    for config_name in sorted(by_config.keys()):
        trials = by_config[config_name]
        amps = {}

        for d in trials:
            j_idx = d["perturbed_joint_idx"]
            tau = d["torques_Nm"]
            ts = d["timestamps_s"]
            freq = d["perturbation"]["frequency_hz"]
            a = np.array([fft_amplitude(tau[:, j], ts, freq) for j in range(n_joints)])
            amps[j_idx] = a

        # Build C_emp
        C = np.zeros((n_joints, n_joints))
        for j, a in amps.items():
            if a[j] > 1e-10:
                C[:, j] = a / a[j]
        C = (C + C.T) / 2.0

        J = np.abs(np.array(trials[0]["theoretical"]["J_matrix"]))
        ii, jj = np.triu_indices(n_joints, k=1)
        c_vals = C[ii, jj]
        j_vals = J[ii, jj]
        d_kin = np.abs(ii - jj).astype(float)
        inv_d = 1.0 / d_kin

        rp, pp = pearsonr(j_vals, c_vals) if np.std(c_vals) > 1e-10 else (0, 1)
        rs, ps = spearmanr(j_vals, c_vals) if np.std(c_vals) > 1e-10 else (0, 1)
        rk, _ = pearsonr(inv_d, c_vals) if np.std(c_vals) > 1e-10 else (0, 1)

        print(f"\n── {config_name} ──")
        print(f"  r(|J|, C_emp)       = {rp:.4f}  (p={pp:.2e})  ← quantum predictor")
        print(f"  r(1/d_kin, C_emp)   = {rk:.4f}               ← kinematic predictor")
        print(f"  Spearman rho        = {rs:.4f}")

        # Detailed pair table
        print(f"\n  {'pair':<8s} │ {'|J_pred|':>8s} │ {'C_emp':>8s} │ {'d_kin':>5s} │ verdict")
        print(f"  {'─' * 8}─┼{'─' * 10}┼{'─' * 10}┼{'─' * 7}┼{'─' * 20}")

        pairs = []
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                pairs.append((J[i, j], C[i, j], abs(i - j), i, j))
        pairs.sort(reverse=True)

        for J_val, C_val, dk, i, j in pairs:
            if J_val > 0.3 and C_val > 0.1:
                verdict = "STRONG coupling OK"
            elif J_val < 0.05 and C_val < 0.05:
                verdict = "weak predicted+measured"
            elif J_val > 0.3 and C_val < 0.05:
                verdict = "!! MISSED coupling"
            elif J_val < 0.05 and C_val > 0.1:
                verdict = "!! UNEXPECTED coupling"
            else:
                verdict = ""
            print(f"  ({i},{j})    │ {J_val:8.4f} │ {C_val:8.4f} │   {dk:>3d} │ {verdict}")

        summary.append({
            "config": config_name,
            "r_coupling": float(rp),
            "p_coupling": float(pp),
            "r_kinematic": float(rk),
            "rho_coupling": float(rs),
            "C_emp": C.tolist(),
            "J_pred": J.tolist(),
        })

    # ── 4. OVERALL SUMMARY ──────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("4. OVERALL SUMMARY")
    print("=" * 85)

    if summary:
        print(f"\n  {'config':<18s} │ {'r(J,C)':>8s} │ {'r(1/d,C)':>8s} │ {'winner':>12s}")
        print(f"  {'─' * 18}─┼{'─' * 10}┼{'─' * 10}┼{'─' * 14}")
        for s in summary:
            winner = "QUANTUM" if abs(s["r_coupling"]) > abs(s["r_kinematic"]) else "kinematic"
            print(f"  {s['config']:<18s} │ {s['r_coupling']:>8.4f} │ {s['r_kinematic']:>8.4f} │ {winner:>12s}")

        r_vals = [s["r_coupling"] for s in summary]
        rk_vals = [s["r_kinematic"] for s in summary]
        print(f"\n  Mean r(|J|, C_emp) = {np.mean(r_vals):.4f}")
        print(f"  Mean r(1/d, C_emp) = {np.mean(rk_vals):.4f}")

        if np.mean(r_vals) > 0.5:
            print(f"\n  *** RESULT: Quantum coupling predictor works! (r={np.mean(r_vals):.3f}) ***")
            if np.mean(r_vals) > np.mean(rk_vals) + 0.1:
                print(f"  *** Quantum predictor BETTER than kinematic by {np.mean(r_vals) - np.mean(rk_vals):.3f} ***")
        elif np.mean(r_vals) > 0.3:
            print(f"\n  *** RESULT: Moderate coupling signal. May need more configs. ***")
        else:
            print(f"\n  *** RESULT: Weak signal. Check sensor data quality. ***")

    # Save
    out_path = Path("results/openarm_diagnostic_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": summary}, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
