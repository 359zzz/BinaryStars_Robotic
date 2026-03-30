#!/usr/bin/env python3
"""Quick diagnostic for Piper coupling experiment results.

Run on the IPC directly — only needs numpy + scipy.

Usage:
    python analysis/quick_diagnostic.py results/
    python analysis/quick_diagnostic.py results/piper_coupling_home_j*.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import signal as sig
from scipy.stats import pearsonr, spearmanr


# ── helpers ─────────────────────────────────────────────────────────────

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
    """Load one JSON trial, return dict with arrays."""
    with open(fpath) as f:
        d = json.load(f)
    d["positions_deg"] = np.array(d["positions_deg"])
    d["commanded_deg"] = np.array(d["commanded_deg"])
    d["timestamps_s"] = np.array(d["timestamps_s"])
    d["_path"] = fpath
    return d


# ── per-trial diagnostic ───────────────────────────────────────────────

def diagnose_trial(d):
    """Return diagnostic dict for one trial."""
    pos = d["positions_deg"]       # (N, 6)
    cmd = d["commanded_deg"]       # (N, 6)
    ts  = d["timestamps_s"]        # (N,)
    j_idx = d["perturbed_joint_idx"]
    freq = d["perturbation"]["frequency_hz"]
    ramp = d["perturbation"]["ramp_s"]
    n_joints = pos.shape[1]

    diag = {
        "file": Path(d["_path"]).name,
        "config": d["config_name"],
        "perturbed": j_idx,
        "n_samples": len(ts),
        "duration_s": float(ts[-1] - ts[0]),
        "joints": [],
    }

    for j in range(n_joints):
        p = pos[:, j]
        c = cmd[:, j]
        dev = p - c
        # Use post-ramp data for stats
        mask = ts >= ramp
        p_ss = p[mask]
        dev_ss = dev[mask]

        jinfo = {
            "joint": j,
            "is_perturbed": j == j_idx,
            "pos_mean": float(np.mean(p)),
            "pos_std": float(np.std(p)),
            "pos_range": float(np.ptp(p)),
            "dev_std": float(np.std(dev_ss)) if len(dev_ss) > 0 else 0.0,
            "dev_max": float(np.max(np.abs(dev_ss))) if len(dev_ss) > 0 else 0.0,
            "fft_amp_pos": fft_amplitude(p, ts, freq, ramp),
            "fft_amp_dev": fft_amplitude(dev, ts, freq, ramp),
        }
        diag["joints"].append(jinfo)

    return diag


# ── per-config coupling analysis ───────────────────────────────────────

def analyze_config_coupling(trials):
    """Build C_emp and compare with J_pred for one config."""
    n_joints = 6
    amps = {}

    for d in trials:
        j_idx = d["perturbed_joint_idx"]
        pos = d["positions_deg"]
        ts = d["timestamps_s"]
        freq = d["perturbation"]["frequency_hz"]
        ramp = d["perturbation"]["ramp_s"]

        a = np.array([
            fft_amplitude(pos[:, j], ts, freq, ramp)
            for j in range(n_joints)
        ])
        amps[j_idx] = a

    # Build C_emp
    C = np.zeros((n_joints, n_joints))
    for j, a in amps.items():
        if a[j] > 1e-10:
            C[:, j] = a / a[j]
    C = (C + C.T) / 2.0

    # Get J_pred from any trial
    J = np.array(trials[0]["theoretical"]["J_matrix"])

    # Correlation (upper triangle)
    ii, jj = np.triu_indices(n_joints, k=1)
    c_vals = C[ii, jj]
    j_vals = np.abs(J[ii, jj])
    d_kin = np.abs(ii - jj).astype(float)
    inv_d = 1.0 / d_kin

    stats = {"n_pairs": len(c_vals)}
    if np.std(c_vals) > 1e-10 and np.std(j_vals) > 1e-10:
        rp, pp = pearsonr(j_vals, c_vals)
        rs, ps = spearmanr(j_vals, c_vals)
        stats["r_coupling"] = float(rp)
        stats["p_coupling"] = float(pp)
        stats["rho_coupling"] = float(rs)
    else:
        stats["r_coupling"] = 0.0
        stats["note"] = "zero variance in C_emp — no coupling signal detected"

    if np.std(c_vals) > 1e-10:
        rk, _ = pearsonr(inv_d, c_vals)
        stats["r_kinematic"] = float(rk)
    else:
        stats["r_kinematic"] = 0.0

    return C, J, stats


# ── main ────────────────────────────────────────────────────────────────

def main():
    # Collect files
    args = sys.argv[1:]
    if not args:
        print("Usage: python quick_diagnostic.py results/")
        sys.exit(1)

    files = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files.extend(sorted(p.glob("piper_coupling_*.json")))
        elif p.is_file() and p.suffix == ".json":
            files.append(p)
    files = sorted(set(files))

    if not files:
        print("No piper_coupling_*.json files found.")
        sys.exit(1)

    print(f"Found {len(files)} trial files\n")

    # ── Load all ───────────────────────────────────────────────────────
    by_config = defaultdict(list)
    all_diags = []

    for f in files:
        d = load_trial(f)
        by_config[d["config_name"]].append(d)
        diag = diagnose_trial(d)
        all_diags.append(diag)

    # ── Per-trial table ────────────────────────────────────────────────
    print("=" * 80)
    print("PER-TRIAL DIAGNOSTIC")
    print("=" * 80)

    for config_name in sorted(by_config.keys()):
        trials = by_config[config_name]
        print(f"\n┌── Config: {config_name} ({len(trials)} trials) ──┐")
        print(f"│  {'file':<32s} │ perturb │ self_amp │ max_other │ other_std │ signal? │")
        print(f"│{'─' * 32}─┼─────────┼──────────┼───────────┼───────────┼─────────│")

        for diag in sorted(all_diags, key=lambda x: x["file"]):
            if diag["config"] != config_name:
                continue
            j_p = diag["perturbed"]
            self_amp = diag["joints"][j_p]["fft_amp_pos"]
            others = [diag["joints"][j]["fft_amp_pos"]
                      for j in range(6) if j != j_p]
            max_other = max(others)
            others_std = [diag["joints"][j]["pos_std"]
                          for j in range(6) if j != j_p]
            max_std = max(others_std)

            # Signal = max_other > 0.01 deg AND > 1% of self
            has_signal = max_other > 0.01 and (max_other / max(self_amp, 1e-10)) > 0.01
            flag = "YES" if has_signal else "no"

            print(f"│  {diag['file']:<32s} │   j{j_p}    │ "
                  f"{self_amp:7.4f}° │  {max_other:7.4f}° │  {max_std:7.4f}° │  {flag:>5s}  │")
        print(f"└{'─' * 78}┘")

    # ── Position variation summary ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("POSITION VARIATION (non-perturbed joints)")
    print("=" * 80)
    print("This tells us whether ANY joint moved when it wasn't being perturbed.\n")

    all_other_stds = []
    all_other_maxdev = []
    all_other_fft = []

    for diag in all_diags:
        j_p = diag["perturbed"]
        for jinfo in diag["joints"]:
            if not jinfo["is_perturbed"]:
                all_other_stds.append(jinfo["pos_std"])
                all_other_maxdev.append(jinfo["dev_max"])
                all_other_fft.append(jinfo["fft_amp_pos"])

    all_other_stds = np.array(all_other_stds)
    all_other_maxdev = np.array(all_other_maxdev)
    all_other_fft = np.array(all_other_fft)

    print(f"  Total non-perturbed joint readings: {len(all_other_stds)}")
    print(f"  Position std:    mean={np.mean(all_other_stds):.6f}°, "
          f"max={np.max(all_other_stds):.6f}°, "
          f"median={np.median(all_other_stds):.6f}°")
    print(f"  Max |deviation|: mean={np.mean(all_other_maxdev):.6f}°, "
          f"max={np.max(all_other_maxdev):.6f}°")
    print(f"  FFT at 0.5Hz:    mean={np.mean(all_other_fft):.6f}°, "
          f"max={np.max(all_other_fft):.6f}°")

    n_with_signal = np.sum(all_other_fft > 0.01)
    print(f"\n  Joints with FFT > 0.01°: {n_with_signal}/{len(all_other_fft)}")
    n_with_std = np.sum(all_other_stds > 0.05)
    print(f"  Joints with std > 0.05°: {n_with_std}/{len(all_other_stds)}")

    if np.max(all_other_fft) < 0.01:
        print("\n  *** VERDICT: No coupling signal detected in position data. ***")
        print("  *** Piper's position controller is too stiff for position-based coupling. ***")
    elif n_with_signal < 5:
        print("\n  *** VERDICT: Marginal signal — only a few joints show any response. ***")
    else:
        print("\n  *** VERDICT: Coupling signal detected! Proceeding with analysis. ***")

    # ── Per-config coupling analysis ───────────────────────────────────
    print("\n" + "=" * 80)
    print("COUPLING ANALYSIS (C_emp vs |J_pred|)")
    print("=" * 80)

    summary = []
    for config_name in sorted(by_config.keys()):
        trials = by_config[config_name]
        C_emp, J_pred, stats = analyze_config_coupling(trials)

        print(f"\n── {config_name} ──")
        print(f"  r(|J|, C_emp)    = {stats['r_coupling']:.4f}")
        print(f"  r(1/d_kin, C_emp)= {stats.get('r_kinematic', 0):.4f}")
        if "rho_coupling" in stats:
            print(f"  Spearman rho     = {stats['rho_coupling']:.4f}")
        if "note" in stats:
            print(f"  NOTE: {stats['note']}")

        # Show top predicted vs measured
        print(f"\n  {'pair':<8s} │ {'|J_pred|':>8s} │ {'C_emp':>8s} │ {'match?':>6s}")
        print(f"  {'─' * 8}─┼{'─' * 10}┼{'─' * 10}┼{'─' * 8}")
        pairs = []
        for i in range(6):
            for j in range(i + 1, 6):
                pairs.append((abs(J_pred[i, j]), C_emp[i, j], i, j))
        pairs.sort(reverse=True)
        for J_val, C_val, i, j in pairs:
            match = "✓" if (J_val > 0.1 and C_val > 0.1) or (J_val < 0.05 and C_val < 0.05) else "✗"
            print(f"  ({i},{j})    │ {J_val:8.4f} │ {C_val:8.4f} │   {match}")

        summary.append({
            "config": config_name,
            "n_trials": len(trials),
            "r_coupling": stats["r_coupling"],
            "r_kinematic": stats.get("r_kinematic", 0),
            "rho_coupling": stats.get("rho_coupling", 0),
            "C_emp": C_emp.tolist(),
            "J_pred": J_pred.tolist(),
        })

    # ── Overall verdict ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    if summary:
        r_vals = [s["r_coupling"] for s in summary]
        print(f"\n  Configs analyzed: {len(summary)}")
        print(f"  r(|J|, C_emp) per config: {', '.join(f'{r:.3f}' for r in r_vals)}")
        print(f"  Mean r = {np.mean(r_vals):.3f}")

        if np.mean(r_vals) > 0.5:
            print("\n  RESULT: Position coupling correlates with J_ij prediction.")
        elif np.max(all_other_fft) < 0.01:
            print("\n  RESULT: Piper position readout cannot resolve coupling.")
            print("  This is NOT a failure — it means the controller is very stiff.")
            print("  Options:")
            print("    1) Use OpenArm (has torque feedback) for the main experiment")
            print("    2) Try Piper with high_follow=True (might be more compliant)")
            print("    3) Report as 'position coupling below detection threshold'")
        else:
            print("\n  RESULT: Weak or inconsistent coupling signal.")

    # ── Save summary ───────────────────────────────────────────────────
    out_path = Path("results/piper_diagnostic_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "n_files": len(files),
        "configs": summary,
        "non_perturbed_stats": {
            "pos_std_mean": float(np.mean(all_other_stds)),
            "pos_std_max": float(np.max(all_other_stds)),
            "fft_mean": float(np.mean(all_other_fft)),
            "fft_max": float(np.max(all_other_fft)),
            "n_with_fft_signal": int(n_with_signal),
            "total_non_perturbed": len(all_other_fft),
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
