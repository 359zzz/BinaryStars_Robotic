#!/usr/bin/env python3
"""Generate coupling heatmap animation from fingerprint trajectory data.

Produces:
  1. Per-motion animated GIF: coupling heatmap evolving over time
  2. Combined evolution PDF: cleaner time-series plot
  3. Per-motion keyframe PDF: 4 snapshots with upper-triangle heatmaps

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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsreal.robot_data.openarm_data import make_openarm_single_arm_ir
from bsreal.dynamics.mass_matrix import compute_mass_matrix
from bsreal.dynamics.coupling import normalized_coupling_matrix

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "figure.dpi": 150,
    "font.family": "sans-serif",
})

JOINT_LABELS = [f"j{i}" for i in range(1, 8)]
N_JOINTS = 7


def compute_coupling_trajectory(positions_deg, timestamps, subsample=5):
    """Compute |J_ij| at subsampled timesteps."""
    ir = make_openarm_single_arm_ir()
    indices = list(range(0, len(timestamps), subsample))

    t_out, J_out, q_out = [], [], []
    for idx in indices:
        q_rad = np.radians(np.array(positions_deg[idx]))
        M = compute_mass_matrix(ir, q_rad)
        J = np.abs(normalized_coupling_matrix(M))
        np.fill_diagonal(J, np.nan)  # mask diagonal
        t_out.append(timestamps[idx])
        J_out.append(J)
        q_out.append(np.array(positions_deg[idx]))

    return np.array(t_out), np.array(J_out), np.array(q_out)


def make_upper_triangle_mask(n):
    """Mask for lower triangle (keep upper + diagonal)."""
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i):
            mask[i, j] = True
    return mask


# ── 1. Animated GIF ────────────────────────────────────────────────────

def generate_gif(motion_data):
    """Generate animated GIF with coupling heatmap + joint angle bar."""
    ts_all = motion_data["timestamps_s"]
    pos_all = motion_data["positions_deg"]
    name = motion_data["motion_name"]

    ts, Js, qs = compute_coupling_trajectory(pos_all, ts_all, subsample=3)
    mask = make_upper_triangle_mask(N_JOINTS)

    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(1, 2, width_ratios=[3, 2], wspace=0.35, figure=fig)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # Initial frame
    J0 = np.ma.array(Js[0], mask=mask)
    im = ax_heat.imshow(J0, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax_heat.set_xticks(range(N_JOINTS))
    ax_heat.set_yticks(range(N_JOINTS))
    ax_heat.set_xticklabels(JOINT_LABELS, fontsize=8)
    ax_heat.set_yticklabels(JOINT_LABELS, fontsize=8)
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="$|J_{ij}|$")
    title = ax_heat.set_title(f"t=0.0s", fontsize=10)

    # Joint angle bar chart
    bars = ax_bar.barh(range(N_JOINTS), qs[0], color="steelblue", height=0.6)
    ax_bar.set_yticks(range(N_JOINTS))
    ax_bar.set_yticklabels(JOINT_LABELS, fontsize=8)
    ax_bar.set_xlabel("Angle (deg)")
    ax_bar.set_xlim(-10, 130)
    ax_bar.set_title("Joint angles", fontsize=10)
    ax_bar.invert_yaxis()

    fig.suptitle(f"{name}", fontsize=11, fontweight="bold")

    def update(frame):
        J = np.ma.array(Js[frame], mask=mask)
        im.set_data(J)
        title.set_text(f"t={ts[frame]:.1f}s")
        for bar, val in zip(bars, qs[frame]):
            bar.set_width(val)
        return [im, title] + list(bars)

    anim = FuncAnimation(fig, update, frames=len(ts), interval=80, blit=False)

    out = FIGURES / f"fingerprint_{name}.gif"
    anim.save(str(out), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"  GIF saved: {out}")


# ── 2. Keyframe PDF (cleaner) ──────────────────────────────────────────

def plot_keyframes(motion_data, n_frames=4):
    """Clean keyframe plot with upper-triangle heatmaps + annotations."""
    ts_all = motion_data["timestamps_s"]
    pos_all = motion_data["positions_deg"]
    name = motion_data["motion_name"]
    desc = motion_data["description"]

    ts, Js, qs = compute_coupling_trajectory(pos_all, ts_all, subsample=3)
    mask = make_upper_triangle_mask(N_JOINTS)

    # Pick evenly spaced frames
    indices = np.linspace(0, len(ts) - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(3.2 * n_frames, 3.2))

    for panel, fi in enumerate(indices):
        ax = axes[panel]
        J = np.ma.array(Js[fi], mask=mask)

        im = ax.imshow(J, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(N_JOINTS))
        ax.set_yticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_LABELS, fontsize=7)
        ax.set_yticklabels(JOINT_LABELS, fontsize=7)

        # Annotate strong couplings (|J| > 0.4)
        for i in range(N_JOINTS):
            for j in range(i + 1, N_JOINTS):
                val = Js[fi][i, j]
                if val > 0.3:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val > 0.7 else "black",
                            fontweight="bold")

        # Title: time + most-moved joint
        q = qs[fi]
        max_j = np.argmax(np.abs(q))
        ax.set_title(f"t={ts[fi]:.1f}s  j{max_j+1}={q[max_j]:.0f}°",
                      fontsize=9, pad=4)

    fig.suptitle(f"{name}: {desc}", fontsize=10, y=1.02)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="$|J_{ij}|$")
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])

    out = FIGURES / f"fingerprint_{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF saved: {out}")


# ── 3. Evolution plot (cleaner) ────────────────────────────────────────

def plot_evolution(all_motions):
    """Time series of top coupling pairs with cleaner styling."""
    pair_data = {}
    for i in range(N_JOINTS):
        for j in range(i + 1, N_JOINTS):
            pair_data[(i, j)] = {"t": [], "J": []}

    t_offset = 0.0
    boundaries, labels = [], []

    for motion in all_motions:
        ts, Js, qs = compute_coupling_trajectory(
            motion["positions_deg"], motion["timestamps_s"], subsample=8)
        labels.append(motion["motion_name"].replace("_", "\n"))
        boundaries.append(t_offset)

        for k in range(len(ts)):
            for i in range(N_JOINTS):
                for j in range(i + 1, N_JOINTS):
                    pair_data[(i, j)]["t"].append(ts[k] + t_offset)
                    pair_data[(i, j)]["J"].append(Js[k][i, j])

        t_offset += ts[-1] + 1.5

    # Top-5 by variation
    ranked = sorted(
        [(np.ptp(d["J"]), i, j) for (i, j), d in pair_data.items() if d["J"]],
        reverse=True,
    )
    top_pairs = [(i, j) for _, i, j in ranked[:5]]

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for idx, (i, j) in enumerate(top_pairs):
        d = pair_data[(i, j)]
        ax.plot(d["t"], d["J"], color=colors[idx], linewidth=1.5,
                label=f"$J_{{{i+1},{j+1}}}$", alpha=0.9)

    for t_b, label in zip(boundaries, labels):
        ax.axvline(t_b, color="#cccccc", linestyle="-", linewidth=0.8)
        ax.text(t_b + 0.5, 1.02, label, fontsize=7, va="bottom",
                color="#666666", transform=ax.get_xaxis_transform())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$|J_{ij}|$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=5, loc="lower right",
              framealpha=0.9, columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIGURES / "fingerprint_evolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Evolution PDF saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    fpath = sys.argv[1] if len(sys.argv) > 1 else "results/fingerprint_trajectory.json"
    with open(fpath) as f:
        data = json.load(f)

    print(f"Motions: {len(data['motions'])}\n")

    for motion in data["motions"]:
        name = motion["motion_name"]
        print(f"Processing: {name} ({motion['n_samples']} samples)")
        plot_keyframes(motion)
        generate_gif(motion)

    print(f"\nGenerating evolution plot...")
    plot_evolution(data["motions"])

    print("\nAll figures generated.")
    print(f"  GIFs in: {FIGURES}/fingerprint_*.gif")
    print(f"  PDFs in: {FIGURES}/fingerprint_*.pdf")


if __name__ == "__main__":
    main()
