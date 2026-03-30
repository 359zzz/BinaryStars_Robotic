#!/usr/bin/env python3
"""Diagnostic for dual-arm Lemma 3 (cross-arm zero coupling) results.

Usage:
    python analysis/lemma3_diagnostic.py results/dual_arm_lemma3.json
"""

import json
import sys
import math
from pathlib import Path

import numpy as np


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


def main():
    fpath = sys.argv[1] if len(sys.argv) > 1 else "results/dual_arm_lemma3.json"
    with open(fpath) as f:
        data = json.load(f)

    print(f"Experiment: {data['experiment']}")
    print(f"Robot: {data['robot']}")
    print(f"Trials: {len(data['trials'])}\n")

    print("=" * 80)
    print("LEMMA 3: CROSS-ARM ZERO COUPLING VERIFICATION")
    print("=" * 80)
    print()
    print("Theory: M_ij = 0 for joints on disjoint subtrees (left vs right arm).")
    print("Prediction: perturbing one arm produces ZERO torque at 0.5 Hz on the other.\n")

    all_own = []
    all_cross = []

    for trial in data["trials"]:
        arm = trial["perturb_arm"]
        j_idx = trial["perturb_joint_idx"]
        jn = trial["perturb_joint_name"]
        n_samples = trial["n_samples"]
        ts = np.array(trial["timestamps_s"])
        own_tau = np.array(trial["own_arm_torques_Nm"])    # (N, 7)
        other_tau = np.array(trial["other_arm_torques_Nm"])  # (N, 7)
        freq = trial["perturbation"]["frequency_hz"]

        print(f"── Perturb: {jn} ({arm} arm, idx={j_idx}), {n_samples} samples ──")

        # Own arm: FFT amplitudes at perturbation frequency
        own_amps = [fft_amplitude(own_tau[:, j], ts, freq) for j in range(7)]
        other_amps = [fft_amplitude(other_tau[:, j], ts, freq) for j in range(7)]

        own_joints = trial["own_arm_joints"]
        other_joints = trial["other_arm_joints"]

        print(f"\n  OWN ARM (should have coupling signal):")
        print(f"  {'joint':<20s} │ {'FFT@0.5Hz':>10s} │ {'std(τ)':>10s}")
        print(f"  {'─' * 20}─┼{'─' * 12}┼{'─' * 12}")
        for j in range(7):
            mask = ts >= 2.0
            std_tau = np.std(own_tau[mask, j])
            marker = " <<<" if j == j_idx else ""
            print(f"  {own_joints[j]:<20s} │ {own_amps[j]:10.5f} │ {std_tau:10.5f}{marker}")

        max_own = max(own_amps[j] for j in range(7) if j != j_idx)

        print(f"\n  OTHER ARM (should be ~noise floor):")
        print(f"  {'joint':<20s} │ {'FFT@0.5Hz':>10s} │ {'std(τ)':>10s}")
        print(f"  {'─' * 20}─┼{'─' * 12}┼{'─' * 12}")
        for j in range(7):
            mask = ts >= 2.0
            std_tau = np.std(other_tau[mask, j])
            print(f"  {other_joints[j]:<20s} │ {other_amps[j]:10.5f} │ {std_tau:10.5f}")

        max_cross = max(other_amps)
        self_amp = own_amps[j_idx]

        ratio = self_amp / max(max_cross, 1e-10)
        print(f"\n  Self amplitude:      {self_amp:.5f} Nm")
        print(f"  Max own-arm other:   {max_own:.5f} Nm")
        print(f"  Max cross-arm:       {max_cross:.5f} Nm")
        print(f"  Self / cross ratio:  {ratio:.1f}x")

        if ratio > 10:
            print(f"  >>> PASS: cross-arm response is {ratio:.0f}x below self (noise floor)")
        elif ratio > 3:
            print(f"  >>> MARGINAL: cross-arm response visible but {ratio:.1f}x below self")
        else:
            print(f"  >>> FAIL: cross-arm response too large (ratio only {ratio:.1f}x)")

        all_own.append(self_amp)
        all_cross.append(max_cross)
        print()

    # Overall summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    own_mean = np.mean(all_own)
    cross_mean = np.mean(all_cross)
    overall_ratio = own_mean / max(cross_mean, 1e-10)

    print(f"\n  Trials:             {len(data['trials'])}")
    print(f"  Mean self amp:      {own_mean:.5f} Nm")
    print(f"  Mean cross-arm amp: {cross_mean:.5f} Nm")
    print(f"  Overall ratio:      {overall_ratio:.1f}x")

    if overall_ratio > 10:
        print(f"\n  *** LEMMA 3 VERIFIED: cross-arm coupling is {overall_ratio:.0f}x below ***")
        print(f"  *** within-arm signal. Consistent with M_ij = 0 for disjoint subtrees. ***")
    elif overall_ratio > 3:
        print(f"\n  *** LEMMA 3 PARTIALLY SUPPORTED: ratio = {overall_ratio:.1f}x ***")
    else:
        print(f"\n  *** LEMMA 3 NOT SUPPORTED: ratio only {overall_ratio:.1f}x ***")

    # Save
    summary = {
        "n_trials": len(data["trials"]),
        "own_arm_amps": all_own,
        "cross_arm_amps": all_cross,
        "overall_ratio": overall_ratio,
        "lemma3_verified": overall_ratio > 10,
    }
    out = Path("results/lemma3_diagnostic_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
