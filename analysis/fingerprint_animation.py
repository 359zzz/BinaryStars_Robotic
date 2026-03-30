#!/usr/bin/env python3
"""Generate coupling heatmap animation from fingerprint trajectory data.

Usage:
    python analysis/fingerprint_animation.py results/fingerprint_trajectory.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.robot_data.openarm_data import make_openarm_single_arm_ir
from bsreal.dynamics.mass_matrix import compute_mass_matrix
from bsreal.dynamics.coupling import normalized_coupling_matrix

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "figure.dpi": 150})


def compute_coupling_trajectory(positions_deg, timestamps, subsample=10):
    """Compute J_ij at subsampled timesteps along trajectory."""
    ir = make_openarm_single_arm_ir()
    n = len(timestamps)
    indices = list(range(0, n, subsample))

    t_out = []
    J_out = []
    q_out = []

    for idx in indices:
        q_deg = np.array(positions_deg[idx])
        q_rad = np.radians(q_deg)
        M = compute_mass_matrix(ir, q_rad)
        J = normalized_coupling_matrix(M)
        t_out.append(timestamps[idx])
        J_out.append(np.abs(J))
        q_out.append(q_deg)

    return np.array(t_out), np.array(J_out), np.array(q_out)


def plot_keyframes(motion_data, n_frames=4):
    """Plot key frames: q(t) + coupling heatmap side by side."""
    ts_all = motion_data["timestamps_s"]
    pos_all = motion_data["positions_deg"]
    name = motion_data["motion_name"]

    ts, Js, qs = compute_coupling_trajectory(pos_all, ts_all, subsample=5)

    # Pick evenly spaced key frames
    frame_indices = np.linspace(0, len(ts) - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(4 * n_frames, 4))
    gs = GridSpec(1, n_frames, figure=fig)

    joint_labels = [f"j{i+1}" for i in range(7)]

    for panel_idx, fi in enumerate(frame_indices):
        ax = fig.add_subplot(gs[0, panel_idx])
        J = Js[fi]
        np.fill_diagonal(J, 0)  # zero diagonal for visual clarity

        im = ax.imshow(J, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(joint_labels, fontsize=7)
        ax.set_yticklabels(joint_labels, fontsize=7)

        t = ts[fi]
        q = qs[fi]
        # Find the joint with largest deviation from 0
        max_j = np.argmax(np.abs(q))
        ax.set_title(f"t={t:.1f}s\nj{max_j+1}={q[max_j]:.0f}°", fontsize=8)

    fig.suptitle(f"Coupling Fingerprint: {name}", fontsize=11, y=1.02)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="$|J_{ij}|$")
    fig.tight_layout(rect=[0, 0, 0.9, 1.0])

    out = FIGURES / f"fingerprint_{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_coupling_evolution(all_motions):
    """Plot time series of top coupling pairs across all motions."""
    ir = make_openarm_single_arm_ir()

    # Top pairs at home config
    q0 = np.zeros(7)
    M0 = compute_mass_matrix(ir, q0)
    J0 = normalized_coupling_matrix(M0)

    # Find top-5 most variable pairs across all motions
    pair_data = {}
    for i in range(7):
        for j in range(i + 1, 7):
            pair_data[(i, j)] = {"t": [], "J": []}

    t_offset = 0.0
    motion_boundaries = []
    motion_labels = []

    for motion in all_motions:
        ts_all = motion["timestamps_s"]
        pos_all = motion["positions_deg"]
        ts, Js, qs = compute_coupling_trajectory(pos_all, ts_all, subsample=10)

        motion_labels.append(motion["motion_name"])
        motion_boundaries.append(t_offset)

        for k in range(len(ts)):
            for i in range(7):
                for j in range(i + 1, 7):
                    pair_data[(i, j)]["t"].append(ts[k] + t_offset)
                    pair_data[(i, j)]["J"].append(Js[k][i, j])

        t_offset += ts[-1] + 1.0

    # Find pairs with most variation
    pair_ranges = []
    for (i, j), d in pair_data.items():
        if d["J"]:
            pair_ranges.append((np.ptp(d["J"]), i, j))
    pair_ranges.sort(reverse=True)
    top_pairs = [(i, j) for _, i, j in pair_ranges[:6]]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_pairs)))

    for idx, (i, j) in enumerate(top_pairs):
        d = pair_data[(i, j)]
        ax.plot(d["t"], d["J"], color=colors[idx], linewidth=1.2,
                label=f"({i+1},{j+1})")

    for t_b, label in zip(motion_boundaries, motion_labels):
        ax.axvline(t_b, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(t_b + 0.3, 0.95, label, fontsize=7, rotation=90,
                va="top", transform=ax.get_xaxis_transform(), color="gray")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$|J_{ij}|$")
    ax.set_title("Coupling Evolution During Robot Motion")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()

    out = FIGURES / "fingerprint_evolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    fpath = sys.argv[1] if len(sys.argv) > 1 else "results/fingerprint_trajectory.json"
    with open(fpath) as f:
        data = json.load(f)

    print(f"Motions: {len(data['motions'])}")

    for motion in data["motions"]:
        print(f"\nProcessing: {motion['motion_name']} ({motion['n_samples']} samples)")
        plot_keyframes(motion)

    print(f"\nGenerating coupling evolution plot...")
    plot_coupling_evolution(data["motions"])

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
