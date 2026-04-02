#!/usr/bin/env python3
"""Analysis for dual-arm coordination experiments (Tables 9-10).

Reads per-trial JSON files from results/coordination/, computes:
- Controller ranking table (per task x config)
- RMSE vs S(rho_L) correlation
- Paired t-tests between controllers
- LaTeX table output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(result_dir: Path) -> list[dict]:
    """Load all per-trial JSON files."""
    results = []
    for f in sorted(result_dir.glob("*.json")):
        if f.name == "coordination_summary.json":
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "task" in data and "controller" in data:
                results.append(data)
        except Exception:
            pass
    return results


def group_by(results: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    """Group results by a combination of keys."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r.get(k, "") for k in keys)
        groups[key].append(r)
    return dict(groups)


def compute_ranking_table(results: list[dict]) -> list[dict]:
    """Per-task x config: rank controllers by mean RMSE."""
    groups = group_by(results, ["task", "config"])
    rows = []
    for (task, config), trials in sorted(groups.items()):
        ctrl_groups = defaultdict(list)
        for t in trials:
            ctrl_groups[t["controller"]].append(t["rmse_total"])

        ranked = sorted(ctrl_groups.items(), key=lambda x: np.mean(x[1]))
        for rank, (ctrl, rmses) in enumerate(ranked, 1):
            rows.append({
                "task": task, "config": config, "controller": ctrl,
                "rank": rank,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "n_reps": len(rmses),
                "s_rho_l": trials[0].get("s_rho_l", 0.0),
                "j_cross_max": trials[0].get("j_cross_max", 0.0),
            })
    return rows


def compute_controller_summary(results: list[dict]) -> list[dict]:
    """Table 9: Per-controller mean RMSE across all tasks."""
    ctrl_groups = defaultdict(list)
    for r in results:
        ctrl_groups[r["controller"]].append(r["rmse_total"])

    rows = []
    for ctrl in ["decoupled", "j_coupled", "c_coupled", "s_adaptive"]:
        if ctrl not in ctrl_groups:
            continue
        rmses = ctrl_groups[ctrl]
        rows.append({
            "controller": ctrl,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_median": float(np.median(rmses)),
            "n_trials": len(rmses),
        })
    return rows


def compute_task_controller_table(results: list[dict]) -> list[dict]:
    """Table 10: Per task x controller mean RMSE (averaged over configs)."""
    groups = group_by(results, ["task", "controller"])
    rows = []
    for (task, ctrl), trials in sorted(groups.items()):
        rmses = [t["rmse_total"] for t in trials]
        rows.append({
            "task": task, "controller": ctrl,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "n_trials": len(rmses),
        })
    return rows


def compute_s_rmse_correlation(results: list[dict]) -> dict:
    """Correlation between S(rho_L) and RMSE improvement."""
    from scipy.stats import pearsonr, spearmanr

    # Group by (task, config), compute RMSE gap between decoupled and c_coupled
    groups = group_by(results, ["task", "config"])
    s_values = []
    rmse_gaps = []

    for (task, config), trials in groups.items():
        ctrl_rmse = {}
        s_val = 0.0
        for t in trials:
            ctrl_rmse.setdefault(t["controller"], []).append(t["rmse_total"])
            s_val = t.get("s_rho_l", 0.0)

        if "decoupled" in ctrl_rmse and "c_coupled" in ctrl_rmse:
            dec_mean = np.mean(ctrl_rmse["decoupled"])
            cc_mean = np.mean(ctrl_rmse["c_coupled"])
            gap = dec_mean - cc_mean  # positive = c_coupled better
            s_values.append(s_val)
            rmse_gaps.append(gap)

    result = {"n_pairs": len(s_values)}
    if len(s_values) >= 3:
        s_arr = np.array(s_values)
        gap_arr = np.array(rmse_gaps)
        if np.std(s_arr) > 1e-10 and np.std(gap_arr) > 1e-10:
            r_p, p_p = pearsonr(s_arr, gap_arr)
            r_s, p_s = spearmanr(s_arr, gap_arr)
            result.update({
                "pearson_r": float(r_p), "pearson_p": float(p_p),
                "spearman_rho": float(r_s), "spearman_p": float(p_s),
                "s_values": s_arr.tolist(),
                "rmse_gaps": gap_arr.tolist(),
            })
    return result


def paired_t_test(results: list[dict], ctrl_a: str, ctrl_b: str) -> dict:
    """Paired t-test comparing two controllers across matched conditions."""
    from scipy.stats import ttest_rel

    groups = group_by(results, ["task", "config", "rep"])
    a_vals, b_vals = [], []
    for key, trials in groups.items():
        a_trial = [t for t in trials if t["controller"] == ctrl_a]
        b_trial = [t for t in trials if t["controller"] == ctrl_b]
        if a_trial and b_trial:
            a_vals.append(a_trial[0]["rmse_total"])
            b_vals.append(b_trial[0]["rmse_total"])

    result = {"ctrl_a": ctrl_a, "ctrl_b": ctrl_b, "n_pairs": len(a_vals)}
    if len(a_vals) >= 2:
        t_stat, p_val = ttest_rel(a_vals, b_vals)
        result.update({
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "mean_diff": float(np.mean(np.array(a_vals) - np.array(b_vals))),
        })
    return result


def format_latex_table9(summary: list[dict]) -> str:
    """Format Table 9: Controller summary."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Controller comparison: mean tracking RMSE (deg) across all tasks and configurations.}",
        r"\label{tab:controller-summary}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Controller & RMSE mean & RMSE std & Median & $N$ \\",
        r"\hline",
    ]
    for row in summary:
        ctrl = row["controller"].replace("_", r"\_")
        lines.append(
            f"  {ctrl} & {row['rmse_mean']:.3f} & {row['rmse_std']:.3f} & "
            f"{row['rmse_median']:.3f} & {row['n_trials']} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def format_latex_table10(task_ctrl: list[dict]) -> str:
    """Format Table 10: Per-task controller comparison."""
    tasks = sorted(set(r["task"] for r in task_ctrl))
    ctrls = ["decoupled", "j_coupled", "c_coupled", "s_adaptive"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Mean tracking RMSE (deg) by task and controller.}",
        r"\label{tab:task-controller}",
        r"\begin{tabular}{l" + "c" * len(ctrls) + "}",
        r"\hline",
        r"Task & " + " & ".join(c.replace("_", r"\_") for c in ctrls) + r" \\",
        r"\hline",
    ]
    lookup = {(r["task"], r["controller"]): r for r in task_ctrl}
    for task in tasks:
        vals = []
        for ctrl in ctrls:
            r = lookup.get((task, ctrl))
            if r:
                vals.append(f"{r['rmse_mean']:.3f}")
            else:
                vals.append("---")
        task_fmt = task.replace("_", r"\_")
        lines.append(f"  {task_fmt} & " + " & ".join(vals) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze coordination experiments")
    parser.add_argument("--input-dir", default="results/coordination")
    parser.add_argument("--output", default="results/coordination_analysis.json")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX tables")
    args = parser.parse_args()

    results = load_results(Path(args.input_dir))
    if not results:
        print(f"No results found in {args.input_dir}")
        sys.exit(1)
    print(f"Loaded {len(results)} trial results")

    # Compute analyses
    ranking = compute_ranking_table(results)
    summary = compute_controller_summary(results)
    task_ctrl = compute_task_controller_table(results)
    s_corr = compute_s_rmse_correlation(results)

    # Statistical tests
    tests = []
    for a, b in [("decoupled", "c_coupled"), ("j_coupled", "c_coupled"),
                 ("decoupled", "j_coupled"), ("c_coupled", "s_adaptive")]:
        tests.append(paired_t_test(results, a, b))

    # Print summary
    print("\n=== Controller Summary (Table 9) ===")
    for row in summary:
        print(f"  {row['controller']:15s}  RMSE={row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f}")

    print("\n=== S(rho_L) vs RMSE Gap Correlation ===")
    if "pearson_r" in s_corr:
        print(f"  Pearson r = {s_corr['pearson_r']:.3f} (p={s_corr['pearson_p']:.4f})")
        print(f"  Spearman rho = {s_corr['spearman_rho']:.3f} (p={s_corr['spearman_p']:.4f})")
    else:
        print("  Insufficient data for correlation")

    print("\n=== Paired t-tests ===")
    for t in tests:
        if "p_value" in t:
            sig = "*" if t["p_value"] < 0.05 else ""
            print(
                f"  {t['ctrl_a']:15s} vs {t['ctrl_b']:15s}: "
                f"diff={t['mean_diff']:+.3f}  p={t['p_value']:.4f} {sig}"
            )

    if args.latex:
        print("\n=== LaTeX Table 9 ===")
        print(format_latex_table9(summary))
        print("\n=== LaTeX Table 10 ===")
        print(format_latex_table10(task_ctrl))

    # Save
    analysis = {
        "n_results": len(results),
        "controller_summary": summary,
        "task_controller": task_ctrl,
        "ranking": ranking,
        "s_rmse_correlation": s_corr,
        "paired_tests": tests,
    }
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis: {args.output}")


if __name__ == "__main__":
    main()
