"""
Run experiments for Smart Edge Offloading Framework
=====================================================
- 500 tasks per run
- Terminal shows HW prediction + reason for every decision
- Cloud tasks hit the REAL server at 13.53.132.84
- Saves 3 publication-quality graphs
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edge.edge_executor        import EdgeExecutor
from cloud.cloud_api           import CloudAPI
from edge.decision_engine      import DecisionEngine
from edge.congestion_predictor import CongestionPredictor
from shared.data_models        import IoTTask

NUM_TASKS = 500
QUEUE_MAX = 20.0
DIVIDER   = "─" * 72

COLORS = {
    "edge":   "#2563EB",
    "cloud":  "#F97316",
    "green":  "#16A34A",
    "bg":     "#F8FAFC",
    "grid":   "#E2E8F0",
}


def generate_tasks(n: int) -> list:
    tasks = []
    for i in range(n):
        tasks.append(IoTTask(
            task_id     = i + 1,
            size        = round(random.uniform(1.0, 10.0), 2),
            compute     = round(random.uniform(1.0,  5.0), 2),
            latency_req = round(random.uniform(0.5,  5.0), 2),
        ))
    return tasks


def simulate_queue_series(length: int) -> list:
    """Realistic drifting queue series to seed Holt-Winters."""
    series, val = [], random.uniform(3, 12)
    for _ in range(length):
        val += random.gauss(0, 1.5)
        if random.random() < 0.05:
            val += random.uniform(5, 10)
        val = max(0.0, min(QUEUE_MAX, val))
        series.append(round(val, 1))
    return series


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("\n" + "="*72)
    print(f"  SMART EDGE OFFLOAD — {NUM_TASKS} TASKS")
    print("="*72)

    edge      = EdgeExecutor()
    cloud     = CloudAPI()          # real server + fallback
    de        = DecisionEngine()
    predictor = CongestionPredictor()
    out_dir   = os.path.dirname(os.path.abspath(__file__))

    tasks      = generate_tasks(NUM_TASKS)
    queue_hist = simulate_queue_series(20)

    # result accumulators
    decisions        = []   # "edge" or "cloud" per task
    execution_times  = []
    energies         = []
    violations       = []
    hw_predictions   = []
    task_sizes       = []

    print(f"\n  Running {NUM_TASKS} tasks...\n")

    for idx, task in enumerate(tasks):

        # update queue history
        new_q = max(0, min(QUEUE_MAX,
            queue_hist[-1] + random.gauss(0, 1.5) +
            (random.uniform(4, 8) if random.random() < 0.05 else 0)))
        queue_hist.append(round(new_q, 1))
        edge_queue  = queue_hist[-1]
        cloud_queue = round(random.uniform(0, QUEUE_MAX * 0.5), 1)

        # Holt-Winters prediction
        recent      = queue_hist[-15:]
        predicted_q = predictor.predict_congestion(recent, silent=True)
        hw_predictions.append(predicted_q)
        method = "HW" if len(recent) >= 5 else "fallback"

        # cost-based decision
        edge_est  = edge.estimate(task)
        cloud_est = cloud.estimate(task)
        decision, reason = de.decide_with_reason(
            task, edge_est, cloud_est, edge_queue, cloud_queue)

        # execute
        if decision == "edge":
            result = edge.execute(task)
        else:
            result = cloud.execute(task)   # hits real server or fallback

        decisions.append(decision)
        execution_times.append(result.execution_time)
        energies.append(result.energy)
        violations.append(1 if result.execution_time > task.latency_req else 0)
        task_sizes.append(task.size)

        # terminal output — every 10th task only (50 printed out of 500)
        if (idx + 1) % 10 == 0:
            print(
                f"  Task {task.task_id:>5} | "
                f"size={task.size:>5.2f}MB  compute={task.compute:.2f}  "
                f"lat_req={task.latency_req:.2f}s\n"
                f"             [{method}] pred_queue={predicted_q:>5.2f}  "
                f"edge_q={edge_queue:.1f}  cloud_q={cloud_queue:.1f}\n"
                f"             edge_est={edge_est['delay']:.4f}s  "
                f"cloud_est={cloud_est['delay']:.4f}s\n"
                f"             DECISION -> {decision.upper():<5}  | {reason}\n"
            )

    # ── summary ───────────────────────────────────────────────────────
    n_edge  = decisions.count("edge")
    n_cloud = decisions.count("cloud")
    avg_t   = sum(execution_times) / NUM_TASKS
    avg_e   = sum(energies)        / NUM_TASKS
    viol_pct= sum(violations)      / NUM_TASKS * 100
    hw_arr  = np.array(hw_predictions)

    edge_times  = [t for t, d in zip(execution_times, decisions) if d == "edge"]
    cloud_times = [t for t, d in zip(execution_times, decisions) if d == "cloud"]

    print(f"\n{'='*72}")
    print(f"  RESULTS — {NUM_TASKS} TASKS")
    print(f"{'='*72}")
    print(f"  {'Edge executions':<35} {n_edge:>6}  ({n_edge/NUM_TASKS*100:.1f}%)")
    print(f"  {'Cloud executions':<35} {n_cloud:>6}  ({n_cloud/NUM_TASKS*100:.1f}%)")
    print(f"  {'Avg execution time (s)':<35} {avg_t:>9.4f}")
    print(f"  {'Avg energy (J)':<35} {avg_e:>9.4f}")
    print(f"  {'Latency violations':<35} {viol_pct:>8.1f}%")
    if edge_times:
        print(f"  {'Avg edge time (s)':<35} {sum(edge_times)/len(edge_times):>9.4f}")
    if cloud_times:
        print(f"  {'Avg cloud time (s)':<35} {sum(cloud_times)/len(cloud_times):>9.4f}")
    print(f"  {'HW pred queue  mean/max':<35} "
          f"{hw_arr.mean():>6.2f} / {hw_arr.max():>5.2f}")

    # ── graph 1: execution time over tasks ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    task_ids = list(range(1, NUM_TASKS + 1))
    edge_ids   = [i for i, d in zip(task_ids, decisions) if d == "edge"]
    cloud_ids  = [i for i, d in zip(task_ids, decisions) if d == "cloud"]
    edge_ts    = [t for t, d in zip(execution_times, decisions) if d == "edge"]
    cloud_ts   = [t for t, d in zip(execution_times, decisions) if d == "cloud"]
    ax.scatter(edge_ids,  edge_ts,  color=COLORS["edge"],  s=10,
               alpha=0.7, label=f"Edge ({n_edge})", zorder=3)
    ax.scatter(cloud_ids, cloud_ts, color=COLORS["cloud"], s=10,
               alpha=0.7, label=f"Cloud ({n_cloud})", zorder=3)
    _style(ax, f"Execution Time per Task  (n={NUM_TASKS})",
           "Task ID", "Execution Time (s)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "graph1_execution_time.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Saved: {p}")

    # ── graph 2: HW predicted queue over time ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.plot(task_ids, hw_predictions, color="#7C3AED", lw=1.2,
            alpha=0.85, label="HW Predicted Queue")
    ax.axhline(y=15, color=COLORS["cloud"], lw=1.5, ls="--",
               label="Congestion threshold (15)")
    # shade congested regions
    hw_arr2 = np.array(hw_predictions)
    ax.fill_between(task_ids, hw_arr2, 15,
                    where=(hw_arr2 >= 15),
                    color=COLORS["cloud"], alpha=0.15, label="Congested")
    _style(ax, "Holt-Winters Queue Prediction over Tasks",
           "Task ID", "Predicted Queue Length")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "graph2_hw_predictions.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")

    # ── graph 3: offloading split + avg times ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # pie
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        [n_edge, n_cloud],
        labels=["Edge", "Cloud"],
        colors=[COLORS["edge"], COLORS["cloud"]],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title("Offloading Split", fontsize=13, fontweight="bold", pad=10)

    # bar: avg time edge vs cloud
    ax2 = axes[1]
    cats = ["Edge", "Cloud"]
    vals = [
        sum(edge_times)/len(edge_times)   if edge_times  else 0,
        sum(cloud_times)/len(cloud_times) if cloud_times else 0,
    ]
    bars = ax2.bar(cats, vals,
                   color=[COLORS["edge"], COLORS["cloud"]],
                   width=0.4, zorder=3)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f"{v:.3f}s", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    _style(ax2, "Avg Execution Time: Edge vs Cloud",
           "Location", "Avg Time (s)")

    fig.suptitle(f"Offloading Summary — {NUM_TASKS} Tasks",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(out_dir, "graph3_offload_summary.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")
    print("\n  All done.")


if __name__ == "__main__":
    main()