"""Signal processing for coupling extraction from perturbation data."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig


def extract_coupling_amplitudes(
    torques: np.ndarray,
    timestamps: np.ndarray,
    frequency_hz: float = 0.5,
    ramp_s: float = 2.0,
) -> np.ndarray:
    """Extract torque oscillation amplitude at the perturbation frequency.

    Parameters
    ----------
    torques : (N, n_joints) array of torque readings (Nm)
    timestamps : (N,) array of timestamps (s)
    frequency_hz : perturbation frequency
    ramp_s : discard initial ramp-up period

    Returns
    -------
    amplitudes : (n_joints,) array of torque amplitudes at frequency_hz
    """
    # Discard ramp-up
    mask = timestamps >= ramp_s
    torques = torques[mask]
    timestamps = timestamps[mask]

    if len(torques) < 10:
        return np.zeros(torques.shape[1])

    n_joints = torques.shape[1]
    amplitudes = np.zeros(n_joints)

    # Estimate sample rate
    dt = np.median(np.diff(timestamps))
    fs = 1.0 / dt

    for j in range(n_joints):
        tau = torques[:, j]

        # Detrend (remove gravity/static offset)
        tau = tau - np.mean(tau)

        # Bandpass filter around perturbation frequency
        f_lo = max(frequency_hz * 0.5, 0.1)
        f_hi = min(frequency_hz * 1.5, fs / 2 - 0.1)

        if f_hi <= f_lo:
            # Frequency range invalid, fall back to FFT only
            pass
        else:
            sos = sig.butter(4, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
            tau = sig.sosfiltfilt(sos, tau)

        # FFT to extract amplitude at perturbation frequency
        N = len(tau)
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_vals = np.fft.rfft(tau)
        magnitudes = 2.0 * np.abs(fft_vals) / N

        # Find the bin closest to frequency_hz
        idx = np.argmin(np.abs(freqs - frequency_hz))
        # Take max of 3 bins around peak (for frequency resolution)
        lo_idx = max(0, idx - 1)
        hi_idx = min(len(magnitudes) - 1, idx + 1)
        amplitudes[j] = np.max(magnitudes[lo_idx : hi_idx + 1])

    return amplitudes


def build_empirical_coupling_matrix(
    all_amplitudes: dict[int, np.ndarray],
    n_joints: int,
) -> np.ndarray:
    """Build empirical coupling matrix from per-joint perturbation amplitudes.

    Parameters
    ----------
    all_amplitudes : {perturbed_joint_idx: amplitudes_array} for each trial
    n_joints : number of joints

    Returns
    -------
    C_emp : (n_joints, n_joints) empirical coupling matrix, normalized
    """
    C = np.zeros((n_joints, n_joints))

    for j, amps in all_amplitudes.items():
        if amps[j] > 1e-10:
            C[:, j] = amps / amps[j]
        else:
            C[:, j] = 0.0

    # Symmetrize
    C = (C + C.T) / 2.0
    return C


def compare_with_theory(
    C_emp: np.ndarray,
    J_pred: np.ndarray,
    kinematic_distances: np.ndarray | None = None,
) -> dict:
    """Compare empirical coupling with predicted J_ij.

    Returns dict with correlation statistics.
    """
    from scipy.stats import pearsonr, spearmanr

    n = C_emp.shape[0]
    # Extract upper-triangle pairs
    i_idx, j_idx = np.triu_indices(n, k=1)

    c_vals = C_emp[i_idx, j_idx]
    j_vals = np.abs(J_pred[i_idx, j_idx])

    # Filter out zero-variance cases
    result = {
        "n_pairs": len(c_vals),
        "c_emp_values": c_vals.tolist(),
        "j_pred_values": j_vals.tolist(),
    }

    if np.std(c_vals) > 1e-10 and np.std(j_vals) > 1e-10:
        r_p, p_p = pearsonr(j_vals, c_vals)
        r_s, p_s = spearmanr(j_vals, c_vals)
        result["pearson_r"] = float(r_p)
        result["pearson_p"] = float(p_p)
        result["spearman_rho"] = float(r_s)
        result["spearman_p"] = float(p_s)
    else:
        result["pearson_r"] = 0.0
        result["spearman_rho"] = 0.0

    if kinematic_distances is not None:
        d_vals = kinematic_distances[i_idx, j_idx]
        inv_d = np.where(d_vals > 0, 1.0 / d_vals, 0.0)
        if np.std(inv_d) > 1e-10 and np.std(c_vals) > 1e-10:
            r_k, _ = pearsonr(inv_d, c_vals)
            result["kinematic_pearson_r"] = float(r_k)
        else:
            result["kinematic_pearson_r"] = 0.0

    return result
