"""
forecasting_metrics.py
======================
Quantitative forecasting-accuracy report for the congestion forecaster
(Reviewer 1: "The paper proposes a forecasting algorithm yet never reports
forecasting accuracy metrics — only a qualitative graph. Add the forecasting
accuracy metrics."; Reviewer 3: "comparative analysis with ... ARIMA").

Computes, strictly one-step-ahead and out-of-sample, the standard forecasting
error metrics for four forecasters on the *same* backlog series:

    Holt-Winters (our method)   — CongestionPredictor
    Naive persistence           — NaivePersistencePredictor   (baseline)
    Single exp. smoothing       — SimpleExpSmoothingPredictor  (no-trend ablation)
    ARIMA(1,1,0)                — ARIMAPredictor                (classical)

Metrics per forecaster (mean ± std across seeds):

    MAE   — mean absolute error (s)
    MSE   — mean squared error (s^2)
    RMSE  — root mean squared error (s)
    sMAPE — symmetric mean absolute percentage error (%)
    DirAcc— directional accuracy (%): fraction of slots where the forecaster
            correctly calls the *sign* of the next backlog change.

Why report DirAcc as well as MAE/RMSE? On a mean-reverting queue series, naive
persistence is a hard-to-beat point-forecast baseline, so Holt-Winters does not
necessarily win on MAE/RMSE. Its claimed value is *early directional warning*,
which is what DirAcc measures. Reporting both makes that argument
quantitatively instead of by assertion — and lets the reader see plainly if
directional accuracy is or is not meaningfully above the 50% chance level.

The backlog series evaluated are those produced by the forecaster-neutral
`reactive` policy (its routing does not depend on any forecast), so all four
forecasters are judged on an identical, unbiased series. Both the `step` and
`ramped` traffic models are evaluated, matching run_experiments.py.

Output
------
- forecasting_metrics.csv          : one row per (traffic_model, series,
                                      forecaster) with mean±std over seeds
- forecasting_metrics_per_seed.csv : raw per-seed rows
- fig_forecasting_metrics.png      : MAE/RMSE and directional-accuracy bars
                                      (edge series) per traffic model

Usage
-----
    python forecasting_metrics.py
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

from edge.congestion_predictor import (
    ARIMAPredictor,
    CongestionPredictor,
    NaivePersistencePredictor,
    SimpleExpSmoothingPredictor,
    compute_forecast_metrics,
)
import run_experiments as base

# First 5 seeds (ARIMA is refit at every slot, so the full 15-seed x 2-model x
# 2-series sweep would be slow; 5 seeds already gives error bars). Bump to
# base.SEEDS for the full set if desired.
EVAL_SEEDS       = base.SEEDS[:5]
TRAFFIC_MODELS   = list(getattr(base, "TRAFFIC_MODELS", ["step", "ramped"]))
REFERENCE_POLICY = "reactive"      # forecaster-neutral series
METRIC_KEYS      = ["mae", "mse", "rmse", "smape", "dir_acc"]

FORECASTERS = {
    "holt_winters":       lambda: CongestionPredictor(verbose_init=False),
    "naive_persistence":  lambda: NaivePersistencePredictor(),
    "simple_exp_smooth":  lambda: SimpleExpSmoothingPredictor(),
    "arima_1_1_0":        lambda: ARIMAPredictor(),
}

FORECASTER_LABELS = {
    "holt_winters":      "Holt-Winters (ours)",
    "naive_persistence": "Naive persistence",
    "simple_exp_smooth": "Single exp. smoothing",
    "arima_1_1_0":       "ARIMA(1,1,0)",
}


def rolling_metrics_for(predictor, series: list[float], min_history: int) -> dict:
    """One-step-ahead, out-of-sample forecast for every slot after warm-up,
    then the full metric suite (MAE/MSE/RMSE/sMAPE/DirAcc) via
    compute_forecast_metrics. The predictor only ever sees history[:idx] when
    forecasting slot idx, so this is a strictly causal evaluation."""
    if len(series) <= min_history:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"),
                 "smape": float("nan"), "dir_acc": float("nan"), "count": 0}

    actual, predicted = [], []
    for idx in range(min_history, len(series)):
        history = series[:idx]
        pred = predictor.predict_congestion(history, silent=True)
        actual.append(series[idx])
        predicted.append(pred)

    m = compute_forecast_metrics(actual, predicted)
    return {
        "mae": m["mae"],
        "mse": m["mse"],
        "rmse": m["rmse"],
        "smape": m["smape"],
        "dir_acc": m["directional_accuracy"],
        "count": m["n"],
    }


def collect_series() -> dict[tuple[str, int], dict[str, list[float]]]:
    """Run the reference policy once per (model, seed) and keep its edge/cloud
    backlog traces. These are the series all forecasters are evaluated on."""
    series = {}
    for model in TRAFFIC_MODELS:
        for seed in EVAL_SEEDS:
            print(f"  [collect] {model} seed={seed} ...")
            r = base.run_policy(REFERENCE_POLICY, seed, traffic_model=model)
            series[(model, seed)] = {
                "edge":  r["edge_trace"],
                "cloud": r["cloud_trace"],
            }
    return series


def evaluate() -> list[dict]:
    series = collect_series()
    rows = []
    for model in TRAFFIC_MODELS:
        for series_name in ("edge", "cloud"):
            for fname, factory in FORECASTERS.items():
                for seed in EVAL_SEEDS:
                    predictor = factory()
                    s = series[(model, seed)][series_name]
                    m = rolling_metrics_for(predictor, s, getattr(predictor, "min_history", 3))
                    rows.append({
                        "traffic_model": model,
                        "series": series_name,
                        "forecaster": fname,
                        "seed": seed,
                        "mae": m["mae"],
                        "mse": m["mse"],
                        "rmse": m["rmse"],
                        "smape": m["smape"],
                        "dir_acc": m["dir_acc"],
                        "count": m["count"],
                    })
                    print(
                        f"  [{model:<6} {series_name:<5} {fname:<18} seed={seed}] "
                        f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} DirAcc={m['dir_acc']:.1f}%"
                    )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["traffic_model"], r["series"], r["forecaster"])].append(r)

    summary = []
    for (model, series_name, fname), entries in grouped.items():
        record = {
            "traffic_model": model,
            "series": series_name,
            "forecaster": fname,
            "n_seeds": len(entries),
        }
        for key in METRIC_KEYS:
            vals = [e[key] for e in entries if e[key] == e[key]]  # drop NaN
            record[key] = float(np.mean(vals)) if vals else float("nan")
            record[key + "_std"] = float(np.std(vals, ddof=0)) if vals else float("nan")
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


def print_summary(summary_rows: list[dict]):
    print("\n" + "=" * 96)
    print("  FORECASTING ACCURACY  (one-step-ahead, out-of-sample, mean +/- std over seeds)")
    print("=" * 96)
    for model in TRAFFIC_MODELS:
        for series_name in ("edge", "cloud"):
            print(f"\n  --- {model} bursts | {series_name} backlog series ---")
            print(f"  {'Forecaster':<22} {'MAE':>9} {'MSE':>10} {'RMSE':>9} {'sMAPE%':>9} {'DirAcc%':>9}")
            print("  " + "-" * 74)
            for fname in FORECASTERS:
                r = next((x for x in summary_rows
                          if x["traffic_model"] == model and x["series"] == series_name
                          and x["forecaster"] == fname), None)
                if not r:
                    continue
                print(
                    f"  {FORECASTER_LABELS[fname]:<22} "
                    f"{r['mae']:>9.4f} {r['mse']:>10.5f} {r['rmse']:>9.4f} "
                    f"{r['smape']:>8.2f}% {r['dir_acc']:>8.1f}%"
                )
    print("\n" + "=" * 96)
    print("  Note: HW is not guaranteed to have the lowest MAE/RMSE on a mean-reverting")
    print("  queue (naive persistence is a strong point-forecast baseline). Its claimed")
    print("  edge is DIRECTIONAL ACCURACY. Report the DirAcc numbers exactly as computed —")
    print("  do not round up 'close to 50%' results; a value near chance-level undercuts")
    print("  rather than supports the anticipation claim.")
    print("=" * 96 + "\n")


def save_figure(summary_rows: list[dict], out_dir: str):
    fnames = list(FORECASTERS.keys())
    colors = {"holt_winters": "#CF994F", "naive_persistence": "#B894AD",
              "simple_exp_smooth": "#0F766E", "arima_1_1_0": "#7C3AED"}

    fig, axes = plt.subplots(2, len(TRAFFIC_MODELS), figsize=(6 * len(TRAFFIC_MODELS), 9))
    if len(TRAFFIC_MODELS) == 1:
        axes = axes.reshape(2, 1)
    fig.patch.set_facecolor("white")

    for col, model in enumerate(TRAFFIC_MODELS):
        edge_rows = {r["forecaster"]: r for r in summary_rows
                     if r["traffic_model"] == model and r["series"] == "edge"}
        x = np.arange(len(fnames))
        width = 0.38

        ax = axes[0][col]
        mae   = [edge_rows[f]["mae"]  for f in fnames]
        mae_e = [edge_rows[f]["mae_std"] for f in fnames]
        rmse   = [edge_rows[f]["rmse"] for f in fnames]
        rmse_e = [edge_rows[f]["rmse_std"] for f in fnames]
        ax.bar(x - width/2, mae,  width, yerr=mae_e,  capsize=3, label="MAE",  color="#2563EB", alpha=0.85)
        ax.bar(x + width/2, rmse, width, yerr=rmse_e, capsize=3, label="RMSE", color="#DC2626", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([FORECASTER_LABELS[f] for f in fnames], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Error (s)")
        ax.set_title(f"Point-forecast error — {model} bursts (edge)")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4, axis="y")

        ax = axes[1][col]
        diracc   = [edge_rows[f]["dir_acc"] for f in fnames]
        diracc_e = [edge_rows[f]["dir_acc_std"] for f in fnames]
        bars = ax.bar(x, diracc, 0.6, yerr=diracc_e, capsize=3,
                      color=[colors[f] for f in fnames], alpha=0.9)
        ax.axhline(50.0, color="black", ls=":", lw=1.2, label="chance level (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels([FORECASTER_LABELS[f] for f in fnames], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Directional accuracy (%)")
        ax.set_title(f"Directional accuracy — {model} bursts (edge)")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4, axis="y")
        for b, v in zip(bars, diracc):
            ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width()/2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Forecasting accuracy: Holt-Winters vs naive / SES / ARIMA\n"
                 "(dotted line = 50% chance level on directional accuracy)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_forecasting_metrics.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_forecasting_metrics.png")


def main():
    print("\n" + "=" * 82)
    print("  Forecasting Accuracy Metrics — HW vs naive / SES / ARIMA")
    print(f"  Seeds: {EVAL_SEEDS}  Reference series: {REFERENCE_POLICY} policy")
    print(f"  Traffic models: {TRAFFIC_MODELS}")
    print("=" * 82)

    rows = evaluate()
    summary_rows = summarize(rows)
    print_summary(summary_rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_csv(rows, os.path.join(out_dir, "forecasting_metrics_per_seed.csv"))
    save_csv(summary_rows, os.path.join(out_dir, "forecasting_metrics.csv"))
    save_figure(summary_rows, out_dir)

    print("\n  Forecasting-accuracy report complete.")
    print("  See forecasting_metrics.csv and fig_forecasting_metrics.png\n")


if __name__ == "__main__":
    main()
