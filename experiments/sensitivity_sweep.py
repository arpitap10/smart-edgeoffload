"""
sensitivity_sweep.py
=====================
Hyperparameter sensitivity analysis for the predictive (Holt-Winters) policy.

Reviewers 1, 2, and 4 all asked: were alpha=0.35, beta=0.25, phi=0.88 (and the
asymmetric blend weights 0.75/0.55, and the offload margin delta=0.08) tuned,
optimized, or chosen experimentally - and how sensitive are the results to
these choices?

This script answers that by re-running the *predictive* policy with each
hyperparameter varied one-at-a-time around its paper-reported default, while
holding all others fixed, across the same 3 seeds used in the original
Table II (7, 19, 42) to keep runtime bounded. It reuses run_policy() from
run_experiments.py unchanged - no simulation logic is duplicated here.

Output
------
- sensitivity_results.csv  : one row per (parameter, value, seed)
- sensitivity_summary.csv  : one row per (parameter, value), averaged over seeds
- fig_sensitivity_*.png    : one plot per swept parameter (violation% and
                             avg latency vs. the swept value)

Usage
-----
    python sensitivity_sweep.py
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

from edge.congestion_predictor import CongestionPredictor
from edge.decision_engine import DecisionEngine
import run_experiments as base  # reuses run_policy, TARGET_TASKS, SLOT_SECONDS, etc.

# ─────────────────────────── sweep configuration ──────────────────────────
SWEEP_SEEDS = [7, 19, 42]           # same 3 seeds as the original Table II
POLICY      = "predictive"

# Paper defaults (see congestion_predictor.py / decision_engine.py)
DEFAULTS = {
    "alpha":      0.35,
    "beta":       0.25,
    "phi":        0.88,
    "delta":      0.08,       # offload_margin
    "w_rising":   0.75,
    "w_easing":   0.55,
}

# One-at-a-time grids around each default (kept modest so the whole sweep
# finishes in a reasonable time: 5 parameters x 3 values x 3 seeds = 45 runs).
GRIDS = {
    "alpha":    [0.20, 0.35, 0.50],
    "beta":     [0.15, 0.25, 0.35],
    "phi":      [0.80, 0.88, 0.95],
    "delta":    [0.04, 0.08, 0.12],
    # w_rising/w_easing are swept together as paired scenarios, since the
    # asymmetry between them (not either value alone) is the design choice
    # under test (Eq. 9 in the paper).
    "w_pair":   [(0.60, 0.60), (0.75, 0.55), (0.90, 0.40)],
}

W_PAIR_LABELS = {
    (0.60, 0.60): "symmetric (0.60/0.60)",
    (0.75, 0.55): "paper default (0.75/0.55)",
    (0.90, 0.40): "aggressive (0.90/0.40)",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Sweep runner
# ═══════════════════════════════════════════════════════════════════════════

def build_engine_and_predictor(param: str, value) -> tuple[DecisionEngine, CongestionPredictor]:
    """Constructs a DecisionEngine + CongestionPredictor for one sweep point,
    holding every other hyperparameter at its paper-default value."""
    p = dict(alpha=DEFAULTS["alpha"], beta=DEFAULTS["beta"], phi=DEFAULTS["phi"])
    delta = DEFAULTS["delta"]
    w_rising, w_easing = DEFAULTS["w_rising"], DEFAULTS["w_easing"]

    if param == "alpha":
        p["alpha"] = value
    elif param == "beta":
        p["beta"] = value
    elif param == "phi":
        p["phi"] = value
    elif param == "delta":
        delta = value
    elif param == "w_pair":
        w_rising, w_easing = value
    else:
        raise ValueError(f"Unknown sweep parameter: {param}")

    predictor = CongestionPredictor(
        alpha=p["alpha"], beta=p["beta"], phi=p["phi"], verbose_init=False,
    )
    engine = DecisionEngine(
        offload_margin=delta, w_rising=w_rising, w_easing=w_easing,
    )
    return engine, predictor


def run_sweep() -> list[dict]:
    rows = []
    for param, grid in GRIDS.items():
        for value in grid:
            engine, predictor = build_engine_and_predictor(param, value)
            label = W_PAIR_LABELS[value] if param == "w_pair" else value
            for seed in SWEEP_SEEDS:
                result = base.run_policy(
                    POLICY, seed,
                    decision_engine=engine, hw_predictor=predictor,
                )
                rows.append({
                    "parameter": param,
                    "value": str(label),
                    "seed": seed,
                    "avg_latency": result["avg_latency"],
                    "p95_latency": result["p95_latency"],
                    "avg_energy": result["avg_energy"],
                    "violation_pct": result["violation_pct"],
                    "cloud_pct": result["cloud_pct"],
                })
                print(
                    f"  [{param}={label} seed={seed}] "
                    f"viol={result['violation_pct']:.2f}%  lat={result['avg_latency']:.3f}s"
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["parameter"], r["value"])].append(r)

    summary = []
    for (param, value), entries in grouped.items():
        record = {"parameter": param, "value": value, "n_seeds": len(entries)}
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


def save_plots(summary_rows: list[dict], out_dir: str):
    params = sorted({r["parameter"] for r in summary_rows})
    for param in params:
        entries = [r for r in summary_rows if r["parameter"] == param]
        # keep original grid order rather than alphabetical string order
        if param == "w_pair":
            order = [W_PAIR_LABELS[v] for v in GRIDS["w_pair"]]
        else:
            order = [str(v) for v in GRIDS[param]]
        entries = sorted(entries, key=lambda r: order.index(r["value"]))

        xs = [r["value"] for r in entries]
        viol = [r["violation_pct"] for r in entries]
        viol_err = [r["violation_pct_std"] for r in entries]
        lat = [r["avg_latency"] for r in entries]
        lat_err = [r["avg_latency_std"] for r in entries]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.patch.set_facecolor("white")

        axes[0].errorbar(xs, viol, yerr=viol_err, fmt="-o", color="#DC2626", capsize=4)
        axes[0].set_title(f"Violation % vs {param}")
        axes[0].set_xlabel(param)
        axes[0].set_ylabel("Deadline Violations (%)")
        axes[0].grid(True, ls="--", alpha=0.5)

        axes[1].errorbar(xs, lat, yerr=lat_err, fmt="-o", color="#2563EB", capsize=4)
        axes[1].set_title(f"Avg Latency vs {param}")
        axes[1].set_xlabel(param)
        axes[1].set_ylabel("Avg Latency (s)")
        axes[1].grid(True, ls="--", alpha=0.5)

        for ax in axes:
            ax.tick_params(axis="x", rotation=15)

        fig.suptitle(f"Sensitivity: {param}  (n={SWEEP_SEEDS}, {base.TARGET_TASKS} tasks/run)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig_sensitivity_{param}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: fig_sensitivity_{param}.png")


def main():
    print("\n" + "=" * 82)
    print("  Hyperparameter Sensitivity Sweep — predictive (Holt-Winters) policy")
    print(f"  Seeds: {SWEEP_SEEDS}  Tasks/run: {base.TARGET_TASKS}")
    print("=" * 82)

    rows = run_sweep()
    summary_rows = summarize(rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_csv(rows, os.path.join(out_dir, "sensitivity_results.csv"))
    save_csv(summary_rows, os.path.join(out_dir, "sensitivity_summary.csv"))
    save_plots(summary_rows, out_dir)

    print("\n  Sensitivity sweep complete.")
    print("  See sensitivity_summary.csv and fig_sensitivity_*.png\n")


if __name__ == "__main__":
    main()
