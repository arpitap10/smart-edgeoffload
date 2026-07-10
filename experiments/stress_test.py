"""
stress_test.py
===============
Robustness check under extreme, high-variance burst conditions (Reviewer 4,
Q1-transferability ask: "Include a 'stress test' section evaluating the
system's performance under extreme, high-variance burst conditions, not just
the current workload profiles.").

This reuses run_policy() from run_experiments.py unchanged, only widening the
burst probability/duration and network-degradation ranges that run_policy
already exposes as parameters. No simulation logic is duplicated.

Two scenarios are compared against the paper's original ("normal") config:

  normal   - burst_prob=0.12, duration 7-13 slots, bw 18-36 Mbps (paper config)
  stress   - burst_prob=0.30, duration 15-25 slots, bw 6-18 Mbps
             (roughly 2.5x more frequent, ~2x longer, ~2x worse bandwidth)

All seven policies are run under both scenarios so the *relative* ordering
(does predictive still win, and by how much) can be checked under conditions
well outside the workload the paper was originally tuned/evaluated on.

Usage
-----
    python stress_test.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_experiments as base  # reuses run_policy, POLICIES, C, LABELS, MARKERS

STRESS_SEEDS = [7, 19, 42]  # same 3 seeds as the original Table II

SCENARIOS = {
    "normal": dict(
        burst_prob=0.12, burst_duration_range=(7, 13),
        network_normal_range=(40.0, 110.0), network_degraded_range=(18.0, 36.0),
        rtt_normal_range=(0.025, 0.055), rtt_degraded_range=(0.055, 0.090),
    ),
    "stress": dict(
        burst_prob=0.30, burst_duration_range=(15, 25),
        network_normal_range=(25.0, 70.0), network_degraded_range=(6.0, 18.0),
        rtt_normal_range=(0.040, 0.080), rtt_degraded_range=(0.090, 0.160),
    ),
}


def run_stress_suite() -> list[dict]:
    rows = []
    for scenario_name, params in SCENARIOS.items():
        for seed in STRESS_SEEDS:
            for policy in base.POLICIES:
                result = base.run_policy(policy, seed, **params)
                rows.append({
                    "scenario": scenario_name,
                    "policy": policy,
                    "seed": seed,
                    "avg_latency": result["avg_latency"],
                    "p95_latency": result["p95_latency"],
                    "avg_energy": result["avg_energy"],
                    "violation_pct": result["violation_pct"],
                    "cloud_pct": result["cloud_pct"],
                })
                print(
                    f"  [{scenario_name:<6} {policy:<18} seed={seed}] "
                    f"viol={result['violation_pct']:.2f}%  lat={result['avg_latency']:.3f}s"
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scenario"], r["policy"])].append(r)

    summary = []
    for (scenario, policy), entries in grouped.items():
        record = {"scenario": scenario, "policy": policy, "n_seeds": len(entries)}
        for key in ["avg_latency", "p95_latency", "avg_energy", "violation_pct", "cloud_pct"]:
            vals = [e[key] for e in entries]
            record[key] = float(np.mean(vals))
            record[key + "_std"] = float(np.std(vals, ddof=0))
        summary.append(record)
    return summary


def save_csv(rows: list[dict], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plot(summary_rows: list[dict], out_dir: str):
    policies = base.POLICIES
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(policies))
    width = 0.35

    for ax, (key, ylabel) in zip(axes, [
        ("violation_pct", "Deadline Violations (%)"),
        ("avg_latency",   "Average Latency (s)"),
    ]):
        for i, scenario in enumerate(["normal", "stress"]):
            vals = [next(r[key] for r in summary_rows
                         if r["scenario"] == scenario and r["policy"] == p) for p in policies]
            errs = [next(r[key + "_std"] for r in summary_rows
                         if r["scenario"] == scenario and r["policy"] == p) for p in policies]
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                   label=scenario, alpha=0.85 if scenario == "normal" else 1.0,
                   color="#94A3B8" if scenario == "normal" else "#DC2626")
        ax.set_xticks(x)
        ax.set_xticklabels([base.LABELS[p] for p in policies], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}: normal vs. stress burst conditions")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4, axis="y")

    fig.suptitle(
        "Stress Test: burst_prob 0.12->0.30, duration 7-13->15-25 slots, "
        "bandwidth 18-36->6-18 Mbps",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_stress_test_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_stress_test_comparison.png")


def print_summary(summary_rows: list[dict]):
    print("\n" + "=" * 96)
    print("  STRESS TEST RESULTS  (mean over seeds)")
    print("=" * 96)
    for scenario in ["normal", "stress"]:
        print(f"\n  --- {scenario.upper()} ---")
        hdr = f"  {'Policy':<20} {'AvgLat':>8} {'Viol%':>8} {'Cloud%':>8}"
        print(hdr)
        for p in base.POLICIES:
            r = next(x for x in summary_rows if x["scenario"] == scenario and x["policy"] == p)
            print(f"  {p:<20} {r['avg_latency']:>7.3f}s {r['violation_pct']:>7.2f}% {r['cloud_pct']:>7.1f}%")
    print("\n" + "=" * 96)


def main():
    print("\n" + "=" * 82)
    print("  Stress Test — normal vs. extreme burst conditions, all policies")
    print(f"  Seeds: {STRESS_SEEDS}  Tasks/run: {base.TARGET_TASKS}")
    print("=" * 82)

    rows = run_stress_suite()
    summary_rows = summarize(rows)
    print_summary(summary_rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_csv(rows, os.path.join(out_dir, "stress_test_results.csv"))
    save_csv(summary_rows, os.path.join(out_dir, "stress_test_summary.csv"))
    save_plot(summary_rows, out_dir)

    print("\n  Stress test complete. See stress_test_summary.csv and fig_stress_test_comparison.png\n")


if __name__ == "__main__":
    main()
