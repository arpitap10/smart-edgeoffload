"""
Run experiments for smart edge offloading framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.device_simulator import IoTSimulator
from edge.edge_executor import EdgeExecutor
from edge.congestion_predictor import CongestionPredictor
from cloud.cloud_api import CloudAPI
from main import SmartEdgeOffloadFramework
import time


def run_pipeline():
    """
    Integrate simulator, monitoring, predictor,
    decision engine, and execution.
    """
    print("Starting Smart Edge Offloading Experiment...")

    simulator      = IoTSimulator(num_devices=10)
    edge_executor  = EdgeExecutor()
    cloud_executor = CloudAPI()

    # Keep a reference to the predictor so we can log HW predictions
    predictor = CongestionPredictor()

    framework = SmartEdgeOffloadFramework(
        simulator=simulator,
        edge_executor=edge_executor,
        cloud_executor=cloud_executor
    )
    # Share the same predictor instance so logs match actual decisions
    framework.predictor = predictor

    num_tasks = 50
    results   = []

    print(f"\nRunning {num_tasks} tasks...\n")

    for i in range(num_tasks):
        task = simulator.generate_task()

        # --- FIX: log the Holt-Winters prediction before executing ---
        queue_series = framework.monitor.queue_series[-10:] or [0]
        if len(queue_series) >= 5:
            predicted_q = predictor.predict_congestion(queue_series)
            hw_note = f"HW predicted queue={predicted_q:.2f}"
        else:
            predicted_q = queue_series[-1] if queue_series else 0
            hw_note = f"fallback avg queue={predicted_q:.2f} (need 5+ pts for HW)"

        print(f"Task {i+1:02d} | device={task.device_id} | size={task.task_size:.2f}MB | {hw_note}")

        start_time = time.time()
        result     = framework.run_task(task)
        total_time = time.time() - start_time

        results.append({
            'task_id':         i + 1,
            'device':          task.device_id,
            'task_size':       task.task_size,
            'decision':        result.location,
            'completion_time': result.completion_time,
            'total_time':      total_time,
            'predicted_queue': predicted_q,
        })

        print(f"         -> {result.location.upper()} | "
              f"completion={result.completion_time:.3f}s | "
              f"total={total_time:.3f}s\n")

    analyze_results(results)


def analyze_results(results):
    """Analyze and print experiment results — FIX: was printing everything twice"""
    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS")
    print("=" * 50)

    total_tasks  = len(results)
    edge_tasks   = sum(1 for r in results if r['decision'] == 'edge')
    cloud_tasks  = sum(1 for r in results if r['decision'] == 'cloud')

    print(f"Total tasks       : {total_tasks}")
    print(f"Edge executions   : {edge_tasks}  ({edge_tasks  / total_tasks * 100:.1f}%)")
    print(f"Cloud executions  : {cloud_tasks} ({cloud_tasks / total_tasks * 100:.1f}%)")

    edge_times  = [r['completion_time'] for r in results if r['decision'] == 'edge']
    cloud_times = [r['completion_time'] for r in results if r['decision'] == 'cloud']

    if edge_times:
        print(f"Avg edge completion time  : {sum(edge_times)  / len(edge_times):.3f}s")
    if cloud_times:
        # FIX: this was 0.000s before because CloudAPI returned server's latency field
        # Now it reflects true measured RTT
        print(f"Avg cloud completion time : {sum(cloud_times) / len(cloud_times):.3f}s")

    edge_sizes  = [r['task_size'] for r in results if r['decision'] == 'edge']
    cloud_sizes = [r['task_size'] for r in results if r['decision'] == 'cloud']

    if edge_sizes:
        print(f"Avg edge task size        : {sum(edge_sizes)  / len(edge_sizes):.2f}MB")
    if cloud_sizes:
        print(f"Avg cloud task size       : {sum(cloud_sizes) / len(cloud_sizes):.2f}MB")

    # Holt-Winters summary
    hw_predictions = [r['predicted_queue'] for r in results]
    print(f"\nHolt-Winters queue predictions:")
    print(f"  Min  : {min(hw_predictions):.2f}")
    print(f"  Max  : {max(hw_predictions):.2f}")
    print(f"  Mean : {sum(hw_predictions) / len(hw_predictions):.2f}")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    run_pipeline()