"""
Run experiments for smart edge offloading framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.device_simulator import IoTSimulator
from edge.edge_executor import EdgeExecutor
from cloud.cloud_api import CloudAPI
from main import SmartEdgeOffloadFramework
import time


def run_pipeline():
    """
    Integrate simulator, monitoring, predictor,
    decision engine, and execution.
    """
    print("Starting Smart Edge Offloading Experiment...")

    # Initialize components
    simulator = IoTSimulator(num_devices=10)
    edge_executor = EdgeExecutor()
    cloud_executor = CloudAPI()

    # Create framework
    framework = SmartEdgeOffloadFramework(
        simulator=simulator,
        edge_executor=edge_executor,
        cloud_executor=cloud_executor
    )

    # Run experiment with multiple tasks
    num_tasks = 50
    results = []

    print(f"Running {num_tasks} tasks...")

    for i in range(num_tasks):
        # Generate task
        task = simulator.generate_task()
        print(f"Task {i+1}: Device {task.device_id}, Size {task.task_size}MB")

        # Run task through framework
        start_time = time.time()
        result = framework.run_task(task)
        total_time = time.time() - start_time

        results.append({
            'task_id': i+1,
            'device': task.device_id,
            'task_size': task.task_size,
            'decision': result.location,
            'completion_time': result.completion_time,
            'total_time': total_time
        })

        print(f"  -> Executed on {result.location}, completion time: {result.completion_time:.3f}s")

    # Analyze results
    analyze_results(results)


def analyze_results(results):
    """Analyze and print experiment results"""
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)

    total_tasks = len(results)
    edge_tasks = sum(1 for r in results if r['decision'] == 'edge')
    cloud_tasks = sum(1 for r in results if r['decision'] == 'cloud')

    print(f"Total tasks: {total_tasks}")
    print(f"Edge executions: {edge_tasks} ({edge_tasks/total_tasks*100:.1f}%)")
    print(f"Cloud executions: {cloud_tasks} ({cloud_tasks/total_tasks*100:.1f}%)")

    # Average completion times
    edge_times = [r['completion_time'] for r in results if r['decision'] == 'edge']
    cloud_times = [r['completion_time'] for r in results if r['decision'] == 'cloud']

    if edge_times:
        print(f"Average edge completion time: {sum(edge_times)/len(edge_times):.3f}s")
    if cloud_times:
        print(f"Average cloud completion time: {sum(cloud_times)/len(cloud_times):.3f}s")

    # Task size analysis
    edge_sizes = [r['task_size'] for r in results if r['decision'] == 'edge']
    cloud_sizes = [r['task_size'] for r in results if r['decision'] == 'cloud']

    if edge_sizes:
        print(f"Average edge task size: {sum(edge_sizes)/len(edge_sizes):.2f}MB")
    if cloud_sizes:
        print(f"Average cloud task size: {sum(cloud_sizes)/len(cloud_sizes):.2f}MB")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    run_pipeline()