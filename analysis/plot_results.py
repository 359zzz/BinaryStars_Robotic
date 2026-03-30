#!/usr/bin/env python3
"""Generate publication figures from experiment results."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "figure.dpi": 200})


def plot_coupling_scatter():
    """Scatter plot: |J_ij| vs C_emp for all pairs across configs."""
    analysis_file = RESULTS / "coupling_analysis.json"
    if not analysis_file.exists():
        print("No coupling_analysis.json found. Run analyze_coupling.py first.")
        return

    with open(analysis_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, config in enumerate(data):
        C_emp = np.array(config["C_emp"])
        J_pred = np.array(config["J_pred"])
        n = C_emp.shape[0]
        i_idx, j_idx = np.triu_indices(n, k=1)

        c_vals = C_emp[i_idx, j_idx]
        j_vals = np.abs(J_pred[i_idx, j_idx])

        ax.scatter(j_vals, c_vals, s=30, alpha=0.6, color=colors[idx],
                   label=f"{config['config_name']} (r={config.get('pearson_r', 0):.2f})")

    ax.set_xlabel(r"Predicted $|J_{ij}|$")
    ax.set_ylabel(r"Measured $C_{\mathrm{emp}}(i,j)$")
    ax.set_title("Real Robot: Coupling Prediction vs Measurement")
    ax.legend(fontsize=7)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    fig.tight_layout()
    fig.savefig(FIGURES / "coupling_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES / 'coupling_scatter.pdf'}")


def plot_coupling_heatmaps():
    """Side-by-side heatmaps: predicted J vs measured C_emp."""
    analysis_file = RESULTS / "coupling_analysis.json"
    if not analysis_file.exists():
        return

    with open(analysis_file) as f:
        data = json.load(f)

    for config in data:
        name = config["config_name"]
        C_emp = np.array(config["C_emp"])
        J_pred = np.abs(np.array(config["J_pred"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

        im1 = ax1.imshow(J_pred, cmap="Blues", vmin=0, vmax=1)
        ax1.set_title(f"Predicted |J| ({name})")
        ax1.set_xlabel("Joint j")
        ax1.set_ylabel("Joint i")
        fig.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(C_emp, cmap="Oranges", vmin=0)
        ax2.set_title(f"Measured C_emp ({name})")
        ax2.set_xlabel("Joint j")
        fig.colorbar(im2, ax=ax2, shrink=0.8)

        fig.tight_layout()
        fig.savefig(FIGURES / f"heatmap_{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: heatmap_{name}.pdf")


def plot_cross_arm():
    """Box plot: within-arm vs cross-arm torque amplitudes."""
    fpath = RESULTS / "dual_arm_lemma3.json"
    if not fpath.exists():
        print("No dual_arm_lemma3.json found.")
        return

    with open(fpath) as f:
        data = json.load(f)

    own_amps = []
    other_amps = []

    from bsreal.experiment.signal_processing import extract_coupling_amplitudes

    for trial in data["trials"]:
        own_tau = np.array(trial["own_arm_torques_Nm"])
        other_tau = np.array(trial["other_arm_torques_Nm"])
        ts = np.array(trial["timestamps_s"])
        freq = trial["perturbation"]["frequency_hz"]

        own_a = extract_coupling_amplitudes(own_tau, ts, freq, ramp_s=2.0)
        other_a = extract_coupling_amplitudes(other_tau, ts, freq, ramp_s=2.0)

        own_amps.extend(own_a.tolist())
        other_amps.extend(other_a.tolist())

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.boxplot([own_amps, other_amps], labels=["Within-arm", "Cross-arm"])
    ax.set_ylabel("Torque amplitude at 0.5 Hz (Nm)")
    ax.set_title("Lemma 3: Cross-Arm Zero Coupling")

    ratio = np.mean(own_amps) / max(np.mean(other_amps), 1e-10)
    ax.text(0.95, 0.95, f"Ratio: {ratio:.1f}x",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES / "cross_arm_lemma3.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: cross_arm_lemma3.pdf")


def main():
    plot_coupling_scatter()
    plot_coupling_heatmaps()
    plot_cross_arm()
    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
