"""Cloud Executor - Handles task execution in cloud"""

from shared.data_models import ExecutionResult, IoTTask
from shared.config import Config
import time
import random


class CloudExecutor:
    """Executes tasks in cloud (unlimited capacity, but with network latency)"""

    def __init__(self):
        self.tasks_executed = 0
        self.total_execution_time = 0

    def execute(self, task: IoTTask) -> ExecutionResult:
        """
        Execute task in cloud.

        Latency model:
          - Network RTT  : sampled from Config.CLOUD_LATENCY_MS (±20 % jitter)
          - Compute time : 0.005 s × task_size  (cloud is faster than edge)
        """
        start_time = time.time()

        # FIX: Simulate realistic network round-trip latency (was missing entirely)
        base_latency_s = Config.CLOUD_LATENCY_MS / 1000.0          # 50 ms default
        jitter_s       = base_latency_s * random.uniform(-0.2, 0.2) # ±20 % jitter
        network_delay  = max(0.0, base_latency_s + jitter_s)

        # Compute time at cloud (faster CPUs than edge)
        compute_time = 0.005 * task.task_size

        time.sleep(network_delay + compute_time)

        completion_time = time.time() - start_time
        self.tasks_executed += 1
        self.total_execution_time += completion_time

        return ExecutionResult(
            status="success",
            location="cloud",
            completion_time=completion_time
        )