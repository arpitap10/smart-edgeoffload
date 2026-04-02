import csv
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cloud.cloud_api import CloudAPI
from cloud.executor import CloudExecutor
from edge.congestion_predictor import CongestionPredictor
from edge.decision_engine import DecisionEngine
from edge.edge_executor import EdgeExecutor
from simulator.device_simulator import IoTSimulator


TARGET_TASKS = 500
SLOT_SECONDS = 0.35
POLICIES = ["edge_only", "cloud_only", "threshold", "reactive", "predictive"]
USE_REAL_CLOUD = True
SEEDS = [42] if USE_REAL_CLOUD else [7, 19, 37]
PRINT_EVERY_NTH_TASK = 20


def sample_arrival_rate(slot_idx: int, burst_slots_remaining: int) -> float:
    base = 1.85
    diurnal = 0.45 * np.sin((2 * np.pi * slot_idx) / 40.0)
    burst = 2.2 if burst_slots_remaining > 0 else 0.0
    return max(0.7, base + diurnal + burst)


def sample_network(rng: np.random.Generator, bursty: bool) -> tuple[float, float]:
    if bursty:
        bandwidth = rng.uniform(18.0, 36.0)
        rtt = rng.uniform(0.055, 0.090)
    else:
        bandwidth = rng.uniform(40.0, 110.0)
        rtt = rng.uniform(0.025, 0.055)
    return float(bandwidth), float(rtt)


def init_metrics() -> dict:
    return {
        "tasks": 0,
        "edge_count": 0,
        "cloud_count": 0,
        "latencies": [],
        "energies": [],
        "violations": 0,
        "edge_backlog_trace": [],
        "cloud_backlog_trace": [],
    }


def choose_policy(
    policy: str,
    task,
    decision_engine: DecisionEngine,
    edge_executor: EdgeExecutor,
    cloud_executor: CloudExecutor,
    current_edge_backlog: float,
    current_cloud_backlog: float,
    predicted_edge_backlog: float,
    predicted_cloud_backlog: float,
    bandwidth_mbps: float,
    rtt: float,
) -> tuple[str, dict, dict, str]:
    edge_est = edge_executor.estimate(task, queue_backlog=current_edge_backlog)
    cloud_est = cloud_executor.estimate(
        task,
        queue_backlog=current_cloud_backlog,
        bandwidth_mbps=bandwidth_mbps,
        rtt=rtt,
    )

    if policy == "edge_only":
        return "edge", edge_est, cloud_est, "EDGE_ONLY baseline: forced local execution."
    if policy == "cloud_only":
        return "cloud", edge_est, cloud_est, "CLOUD_ONLY baseline: forced remote execution."
    if policy == "threshold":
        decision = "cloud" if current_edge_backlog > 0.9 or task.size > 5.5 else "edge"
        threshold_reason = (
            f"THRESHOLD baseline: edge_backlog={current_edge_backlog:.3f}s "
            f"task_size={task.size:.2f}MB -> {decision.upper()}"
        )
        return decision, edge_est, cloud_est, threshold_reason
    if policy == "reactive":
        decision, reason, _, _ = decision_engine.decide_with_reason(
            task,
            edge_est,
            cloud_est,
            current_edge_backlog=current_edge_backlog,
            current_cloud_backlog=current_cloud_backlog,
        )
        return decision, edge_est, cloud_est, reason

    decision, reason, _, _ = decision_engine.decide_with_reason(
        task,
        edge_est,
        cloud_est,
        current_edge_backlog=current_edge_backlog,
        current_cloud_backlog=current_cloud_backlog,
        predicted_edge_backlog=predicted_edge_backlog,
        predicted_cloud_backlog=predicted_cloud_backlog,
    )
    return decision, edge_est, cloud_est, reason


def print_task_trace(
    *,
    seed: int,
    policy: str,
    task,
    task_idx: int,
    decision: str,
    reason: str,
    edge_est: dict,
    cloud_est: dict,
    current_edge_backlog: float,
    current_cloud_backlog: float,
    predicted_edge_backlog: float,
    predicted_cloud_backlog: float,
):
    print(
        f"[seed={seed} policy={policy} task={task_idx:03d}] "
        f"HW(edge={predicted_edge_backlog:.3f}s cloud={predicted_cloud_backlog:.3f}s) "
        f"current(edge={current_edge_backlog:.3f}s cloud={current_cloud_backlog:.3f}s) "
        f"est(edge={edge_est['delay']:.3f}s cloud={cloud_est['delay']:.3f}s) "
        f"task(size={task.size:.2f}MB compute={task.compute:.2f} lat={task.latency_req:.2f}s) "
        f"-> {decision.upper()} | {reason}"
    )


def run_policy(policy: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    task_simulator = IoTSimulator(seed=seed)
    edge_executor = EdgeExecutor()
    cloud_executor = CloudAPI(use_remote=USE_REAL_CLOUD)
    decision_engine = DecisionEngine()
    predictor = CongestionPredictor()
    metrics = init_metrics()

    edge_backlog = 0.0
    cloud_backlog = 0.0
    edge_history = [0.0]
    cloud_history = [0.0]
    arrival_history = [0.0]
    recent_edge_service = [0.18]
    recent_cloud_service = [0.08]
    edge_abs_errors = []
    edge_naive_errors = []
    cloud_abs_errors = []
    cloud_naive_errors = []
    burst_slots_remaining = 0

    slot_idx = 0
    while metrics["tasks"] < TARGET_TASKS:
        edge_backlog = max(0.0, edge_backlog - SLOT_SECONDS)
        cloud_backlog = max(0.0, cloud_backlog - SLOT_SECONDS)

        if burst_slots_remaining == 0 and rng.random() < 0.12:
            burst_slots_remaining = int(rng.integers(7, 13))

        arrival_rate = sample_arrival_rate(slot_idx, burst_slots_remaining)
        bursty = burst_slots_remaining > 0
        bandwidth_mbps, rtt = sample_network(rng, bursty)
        arrivals = int(rng.poisson(arrival_rate * SLOT_SECONDS))
        remaining_tasks = TARGET_TASKS - metrics["tasks"]
        arrivals = min(arrivals, remaining_tasks)

        predicted_arrivals = predictor.predict_congestion(arrival_history, silent=True)
        mean_edge_service = float(np.mean(recent_edge_service))
        mean_cloud_service = float(np.mean(recent_cloud_service))
        predicted_edge_backlog = predictor.predict_congestion(edge_history, silent=True) + (
            0.65 * predicted_arrivals * mean_edge_service
        )
        predicted_cloud_backlog = predictor.predict_congestion(cloud_history, silent=True) + (
            0.35 * predicted_arrivals * mean_cloud_service
        )
        naive_edge_backlog = edge_history[-1]
        naive_cloud_backlog = cloud_history[-1]

        for _ in range(arrivals):
            task = task_simulator.generate_task()
            current_edge_backlog = edge_backlog
            current_cloud_backlog = cloud_backlog

            decision, edge_est, cloud_est, reason = choose_policy(
                policy=policy,
                task=task,
                decision_engine=decision_engine,
                edge_executor=edge_executor,
                cloud_executor=cloud_executor,
                current_edge_backlog=current_edge_backlog,
                current_cloud_backlog=current_cloud_backlog,
                predicted_edge_backlog=predicted_edge_backlog,
                predicted_cloud_backlog=predicted_cloud_backlog,
                bandwidth_mbps=bandwidth_mbps,
                rtt=rtt,
            )

            if decision == "edge":
                result = edge_executor.execute(task, queue_backlog=current_edge_backlog)
                edge_backlog += edge_est["service_time"]
                metrics["edge_count"] += 1
                recent_edge_service.append(edge_est["service_time"])
                recent_edge_service = recent_edge_service[-25:]
            else:
                result = cloud_executor.execute(
                    task,
                    queue_backlog=current_cloud_backlog,
                    bandwidth_mbps=bandwidth_mbps,
                    rtt=rtt,
                )
                cloud_backlog += cloud_est["service_time"]
                metrics["cloud_count"] += 1
                recent_cloud_service.append(cloud_est["service_time"])
                recent_cloud_service = recent_cloud_service[-25:]

            metrics["tasks"] += 1
            if metrics["tasks"] % PRINT_EVERY_NTH_TASK == 0:
                print_task_trace(
                    seed=seed,
                    policy=policy,
                    task=task,
                    task_idx=metrics["tasks"],
                    decision=decision,
                    reason=reason,
                    edge_est=edge_est,
                    cloud_est=cloud_est,
                    current_edge_backlog=current_edge_backlog,
                    current_cloud_backlog=current_cloud_backlog,
                    predicted_edge_backlog=predicted_edge_backlog,
                    predicted_cloud_backlog=predicted_cloud_backlog,
                )
            metrics["latencies"].append(result.execution_time)
            metrics["energies"].append(result.energy)
            if result.execution_time > task.latency_req:
                metrics["violations"] += 1

        metrics["edge_backlog_trace"].append(edge_backlog)
        metrics["cloud_backlog_trace"].append(cloud_backlog)
        edge_abs_errors.append(abs(predicted_edge_backlog - edge_backlog))
        edge_naive_errors.append(abs(naive_edge_backlog - edge_backlog))
        cloud_abs_errors.append(abs(predicted_cloud_backlog - cloud_backlog))
        cloud_naive_errors.append(abs(naive_cloud_backlog - cloud_backlog))
        edge_history.append(edge_backlog)
        cloud_history.append(cloud_backlog)
        arrival_history.append(float(arrivals))
        burst_slots_remaining = max(0, burst_slots_remaining - 1)
        slot_idx += 1

    return {
        "policy": policy,
        "seed": seed,
        "tasks": metrics["tasks"],
        "edge_pct": 100.0 * metrics["edge_count"] / max(metrics["tasks"], 1),
        "cloud_pct": 100.0 * metrics["cloud_count"] / max(metrics["tasks"], 1),
        "avg_latency": float(np.mean(metrics["latencies"])),
        "p95_latency": float(np.percentile(metrics["latencies"], 95)),
        "avg_energy": float(np.mean(metrics["energies"])),
        "violation_pct": 100.0 * metrics["violations"] / max(metrics["tasks"], 1),
        "avg_edge_backlog": float(np.mean(metrics["edge_backlog_trace"])),
        "avg_cloud_backlog": float(np.mean(metrics["cloud_backlog_trace"])),
        "edge_pred_mae": float(np.mean(edge_abs_errors)),
        "edge_naive_mae": float(np.mean(edge_naive_errors)),
        "cloud_pred_mae": float(np.mean(cloud_abs_errors)),
        "cloud_naive_mae": float(np.mean(cloud_naive_errors)),
        "edge_trace": metrics["edge_backlog_trace"],
        "cloud_trace": metrics["cloud_backlog_trace"],
    }


def aggregate_results(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)

    summary = []
    for policy in POLICIES:
        entries = grouped[policy]
        summary.append(
            {
                "policy": policy,
                "avg_latency": float(np.mean([r["avg_latency"] for r in entries])),
                "p95_latency": float(np.mean([r["p95_latency"] for r in entries])),
                "avg_energy": float(np.mean([r["avg_energy"] for r in entries])),
                "violation_pct": float(np.mean([r["violation_pct"] for r in entries])),
                "cloud_pct": float(np.mean([r["cloud_pct"] for r in entries])),
                "avg_edge_backlog": float(np.mean([r["avg_edge_backlog"] for r in entries])),
                "avg_cloud_backlog": float(np.mean([r["avg_cloud_backlog"] for r in entries])),
                "edge_pred_mae": float(np.mean([r["edge_pred_mae"] for r in entries])),
                "edge_naive_mae": float(np.mean([r["edge_naive_mae"] for r in entries])),
                "cloud_pred_mae": float(np.mean([r["cloud_pred_mae"] for r in entries])),
                "cloud_naive_mae": float(np.mean([r["cloud_naive_mae"] for r in entries])),
            }
        )
    return summary


def save_csv(rows: list[dict], path: str):
    fieldnames = [key for key in rows[0].keys() if not key.endswith("_trace")]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(serializable)


def save_figures(all_rows: list[dict], summary_rows: list[dict], out_dir: str):
    summary_by_policy = {row["policy"]: row for row in summary_rows}
    policies = POLICIES

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#0F766E", "#B45309", "#4F46E5", "#DC2626", "#2563EB"]

    axes[0].bar(
        policies,
        [summary_by_policy[p]["avg_latency"] for p in policies],
        color=colors,
    )
    axes[0].set_title("Average Latency")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(
        policies,
        [summary_by_policy[p]["violation_pct"] for p in policies],
        color=colors,
    )
    axes[1].set_title("Deadline Violations")
    axes[1].set_ylabel("Percent")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(
        policies,
        [summary_by_policy[p]["cloud_pct"] for p in policies],
        color=colors,
    )
    axes[2].set_title("Cloud Offload Share")
    axes[2].set_ylabel("Percent")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "policy_comparison.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    predictive_rows = [row for row in all_rows if row["policy"] == "predictive"]
    mean_edge_trace = np.mean([row["edge_trace"] for row in predictive_rows], axis=0)
    mean_cloud_trace = np.mean([row["cloud_trace"] for row in predictive_rows], axis=0)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(mean_edge_trace, label="Edge backlog", color="#2563EB", linewidth=2)
    ax.plot(mean_cloud_trace, label="Cloud backlog", color="#F97316", linewidth=2)
    ax.set_title("Predictive Policy Backlog Evolution")
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Backlog (s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "predictive_backlog_trace.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary_rows: list[dict]):
    print("\n===== AGGREGATED RESULTS ACROSS SEEDS =====")
    header = (
        f"{'Policy':<12} {'AvgLat':>8} {'P95':>8} {'Energy':>8} "
        f"{'Viol%':>8} {'Cloud%':>8} {'EdgeMAE':>10} {'CloudMAE':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['policy']:<12} "
            f"{row['avg_latency']:>8.3f} "
            f"{row['p95_latency']:>8.3f} "
            f"{row['avg_energy']:>8.3f} "
            f"{row['violation_pct']:>8.2f} "
            f"{row['cloud_pct']:>8.2f} "
            f"{row['edge_pred_mae']:>10.4f} "
            f"{row['cloud_pred_mae']:>10.4f}"
        )


def main():
    rows = []
    for seed in SEEDS:
        print(f"\n[Experiment] Running seed={seed}")
        for policy in POLICIES:
            row = run_policy(policy, seed)
            rows.append(row)
            print(
                f"  {policy:<12} latency={row['avg_latency']:.3f}s "
                f"viol={row['violation_pct']:.2f}% cloud={row['cloud_pct']:.2f}%"
            )

    summary_rows = aggregate_results(rows)
    print_summary(summary_rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_csv(rows, os.path.join(out_dir, "per_seed_results.csv"))
    save_csv(summary_rows, os.path.join(out_dir, "summary_results.csv"))
    save_figures(rows, summary_rows, out_dir)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
