#!/usr/bin/env python3
"""Publication-quality plots for coordination experiments (Fig 7).

Fig 7b: Grouped bar chart (task x controller -> RMSE)
Fig 7c: Scatter plot S(rho_L) vs RMSE gap (decoupled - c_coupled)
Fig 7d: Time-series comparison for a single trial
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Publication style
rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CTRL_COLORS = {
    "decoupled":  "#377eb8",
    "j_coupled":  "#ff7f00",
    "c_coupled":  "#4daf4a",
    "s_adaptive": "#984ea3",
}
CTRL_LABELS = {
    "decoupled":  "Decoupled",
    "j_coupled":  "J-coupled",
    "c_coupled":  "C-coupled",
    "s_adaptive": "S-adaptive",
}
TASK_LABELS = {
    "independent":  "Independent",
    "box_lift":     "Box lift",
    "barbell_lift": "Barbell lift",
    "rod_rotation": "Rod rotation",
}


def load_analysis(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_fig7b(analysis: dict, output_dir: Path):
    """Grouped bar chart: task x controller -> RMSE."""
    task_ctrl = analysis["task_controller"]
    tasks = list(dict.fromkeys(r["task"] for r in task_ctrl))
    ctrls = ["decoupled", "j_coupled", "c_coupled", "s_adaptive"]
    ctrls = [c for c in ctrls if any(r["controller"] == c for r in task_ctrl)]

    lookup = {(r["task"], r["controller"]): r for r in task_ctrl}

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(len(tasks))
    width = 0.18
    offsets = np.linspace(-(len(ctrls) - 1) * width / 2,
                           (len(ctrls) - 1) * width / 2, len(ctrls))

    for i, ctrl in enumerate(ctrls):
        means = []
        stds = []
        for task in tasks:
            r = lookup.get((task, ctrl))
            means.append(r["rmse_mean"] if r else 0)
            stds.append(r["rmse_std"] if r else 0)

        ax.bar(x + offsets[i], means, width, yerr=stds, capsize=2,
               label=CTRL_LABELS.get(ctrl, ctrl),
               color=CTRL_COLORS.get(ctrl, "#999999"),
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Task")
    ax.set_ylabel("Tracking RMSE (deg)")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=15, ha="right")
    ax.legend(ncol=2, loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "fig7b_rmse_by_task.pdf"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_fig7c(analysis: dict, output_dir: Path):
    """Scatter: S(rho_L) vs RMSE gap (decoupled - c_coupled)."""
    s_corr = analysis.get("s_rmse_correlation", {})
    s_vals = s_corr.get("s_values", [])
    gaps = s_corr.get("rmse_gaps", [])

    if len(s_vals) < 2:
        print("Insufficient data for Fig 7c")
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.scatter(s_vals, gaps, s=30, c="#4daf4a", edgecolors="black",
               linewidths=0.5, zorder=3)

    # Trend line
    s_arr = np.array(s_vals)
    gap_arr = np.array(gaps)
    if len(s_arr) >= 3:
        z = np.polyfit(s_arr, gap_arr, 1)
        p = np.poly1d(z)
        s_sorted = np.sort(s_arr)
        ax.plot(s_sorted, p(s_sorted), "--", color="gray", linewidth=1, alpha=0.7)

    r_val = s_corr.get("pearson_r", 0.0)
    ax.text(0.05, 0.95, f"$r = {r_val:.2f}$",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$S(\rho_L)$")
    ax.set_ylabel(r"$\Delta$RMSE (decoupled $-$ C-coupled, deg)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = output_dir / "fig7c_s_vs_rmse_gap.pdf"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_fig7d(trial_dir: Path, output_dir: Path, task: str = "box_lift",
               config: str = "home", joint_idx: int = 3):
    """Time-series comparison: q_target vs q_actual for different controllers."""
    ctrls = ["decoupled", "j_coupled", "c_coupled", "s_adaptive"]

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.0), sharex=True, sharey=True)

    for ax, ctrl in zip(axes.flat, ctrls):
        # Find trial file (rep 0)
        pattern = f"{task}_{ctrl}_{config}_rep0.json"
        files = list(trial_dir.glob(pattern))
        if not files:
            ax.set_title(CTRL_LABELS.get(ctrl, ctrl))
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        with open(files[0]) as f:
            data = json.load(f)

        ts = np.array(data.get("timestamps", []))
        q_tgt = np.array(data.get("q_target_deg", []))
        q_act = np.array(data.get("q_actual_deg", []))

        if len(ts) < 2 or q_tgt.ndim < 2:
            ax.set_title(CTRL_LABELS.get(ctrl, ctrl))
            continue

        ax.plot(ts, q_tgt[:, joint_idx], "-", color="black", linewidth=1,
                label="Target", alpha=0.7)
        ax.plot(ts, q_act[:, joint_idx], "-",
                color=CTRL_COLORS.get(ctrl, "blue"), linewidth=1, label="Actual")

        rmse = data.get("rmse_total", 0.0)
        ax.set_title(f"{CTRL_LABELS.get(ctrl, ctrl)} (RMSE={rmse:.2f})")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel(f"Joint {joint_idx + 1} (deg)")
    axes[1, 0].set_ylabel(f"Joint {joint_idx + 1} (deg)")

    fig.suptitle(f"Tracking: {TASK_LABELS.get(task, task)}, {config}", fontsize=10)
    fig.tight_layout()
    path = output_dir / f"fig7d_timeseries_{task}_{config}.pdf"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot coordination results")
    parser.add_argument("--analysis", default="results/coordination_analysis.json")
    parser.add_argument("--trial-dir", default="results/coordination")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--task", default="box_lift", help="Task for Fig 7d")
    parser.add_argument("--config", default="home", help="Config for Fig 7d")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = load_analysis(Path(args.analysis))
    plot_fig7b(analysis, output_dir)
    plot_fig7c(analysis, output_dir)
    plot_fig7d(Path(args.trial_dir), output_dir, args.task, args.config)
    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
