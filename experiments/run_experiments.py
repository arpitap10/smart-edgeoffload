import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.edge_executor import EdgeExecutor
from cloud.executor import CloudExecutor
from edge.decision_engine import DecisionEngine
from shared.data_models import IoTTask
import random
import matplotlib.pyplot as plt

NUM_DEVICES = 500
EPISODES = 5

# -------- BASELINE DECISION --------
def baseline_decide(task):
    # simple heuristic
    if task.size * task.compute < 15:
        return "edge"
    else:
        return "cloud"

# -------- TASK GENERATION --------
def generate_tasks(start_id):
    tasks = []
    for i in range(NUM_DEVICES):
        task = IoTTask(
            task_id=start_id + i,
            size=random.uniform(1, 10),
            compute=random.uniform(1, 5),
            latency_req=random.uniform(1, 5)
        )
        tasks.append(task)
    return tasks

def main():
    edge = EdgeExecutor()
    cloud = CloudExecutor()
    decision_engine = DecisionEngine()

    task_counter = 0

    # ----- OUR MODEL METRICS -----
    our_delay = []
    our_energy = []

    # ----- BASELINE METRICS -----
    base_delay = []
    base_energy = []

    for ep in range(EPISODES):
        tasks = generate_tasks(task_counter)
        task_counter += NUM_DEVICES

        edge_queue = random.uniform(0, 20)
        cloud_queue = random.uniform(0, 20)

        total_delay_our = 0
        total_energy_our = 0

        total_delay_base = 0
        total_energy_base = 0

        for task in tasks:
            # ---------- OUR MODEL ----------
            edge_est = edge.estimate(task)
            cloud_est = cloud.estimate(task)

            decision = decision_engine.decide(
                task, edge_est, cloud_est,
                edge_queue, cloud_queue
            )

            if decision == "edge":
                result = edge.execute(task)
            else:
                result = cloud.execute(task)

            total_delay_our += result.execution_time
            total_energy_our += result.energy

            # ---------- BASELINE ----------
            base_decision = baseline_decide(task)

            if base_decision == "edge":
                result_b = edge.execute(task)
            else:
                result_b = cloud.execute(task)

            total_delay_base += result_b.execution_time
            total_energy_base += result_b.energy

        our_delay.append(total_delay_our / NUM_DEVICES)
        our_energy.append(total_energy_our / NUM_DEVICES)

        base_delay.append(total_delay_base / NUM_DEVICES)
        base_energy.append(total_energy_base / NUM_DEVICES)

        print(f"Episode {ep+1} done")

    # ===== GRAPH 1: DELAY COMPARISON =====
    plt.figure()
    plt.plot(our_delay, label="Proposed Model")
    plt.plot(base_delay, label="Baseline")
    plt.legend()
    plt.title("Delay Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay")
    plt.savefig("comparison_delay.png")

    # ===== GRAPH 2: ENERGY COMPARISON =====
    plt.figure()
    plt.plot(our_energy, label="Proposed Model")
    plt.plot(base_energy, label="Baseline")
    plt.legend()
    plt.title("Energy Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average Energy")
    plt.savefig("comparison_energy.png")

    # ===== GRAPH 3: FINAL BAR GRAPH (IMPORTANT 🔥) =====
    labels = ["Delay", "Energy"]

    our_avg = [
        sum(our_delay)/len(our_delay),
        sum(our_energy)/len(our_energy)
    ]

    base_avg = [
        sum(base_delay)/len(base_delay),
        sum(base_energy)/len(base_energy)
    ]

    x = range(len(labels))

    plt.figure()
    plt.bar(x, our_avg, label="Proposed Model")
    plt.bar(x, base_avg, bottom=our_avg, alpha=0.5, label="Baseline")
    plt.xticks(x, labels)
    plt.title("Overall Performance Comparison")
    plt.legend()
    plt.savefig("final_comparison.png")

    print("\nSaved graphs:")
    print("comparison_delay.png")
    print("comparison_energy.png")
    print("final_comparison.png")

if __name__ == "__main__":
    main()