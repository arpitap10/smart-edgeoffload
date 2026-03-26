"""Edge Executor - Handles task execution at edge nodes"""

from shared.data_models import ExecutionResult, IoTTask
from shared.config import Config
import time


class EdgeExecutor:
    """Executes tasks on edge nodes"""

    def __init__(self,
                 cpu_capacity: int = Config.EDGE_CPU_CAPACITY,
                 memory_gb: int = Config.EDGE_MEMORY_GB):
        self.cpu_capacity = cpu_capacity
        self.memory_gb = memory_gb
        self.current_cpu_usage = 0  # percentage (0–100), decays after each task
        self.tasks_executed = 0
        self.total_execution_time = 0

    def _task_cpu_percent(self, task: IoTTask) -> float:
        """Return the CPU % a task consumes (0–100 scale per core)."""
        # task_size is 1–10 MB; map linearly to 10–100 % of one core
        return (task.task_size / 10.0) * 100.0

    def can_execute(self, task: IoTTask) -> bool:
        """
        Check if edge can accept the task without exceeding total CPU capacity.
        Total capacity = cpu_capacity cores × 100 % each.
        """
        required_cpu = self._task_cpu_percent(task)
        total_capacity = self.cpu_capacity * 100.0
        return (self.current_cpu_usage + required_cpu) <= total_capacity

    def execute(self, task: IoTTask) -> ExecutionResult:
        """Execute task on edge and release CPU when done."""
        start_time = time.time()

        required_cpu = self._task_cpu_percent(task)

        # Acquire CPU slice
        self.current_cpu_usage += required_cpu

        # Simulate execution time proportional to task size
        execution_time = 0.01 * task.task_size  # seconds
        time.sleep(execution_time)

        # FIX: Release CPU after task completes so capacity doesn't fill up forever
        self.current_cpu_usage = max(0.0, self.current_cpu_usage - required_cpu)

        completion_time = time.time() - start_time
        self.tasks_executed += 1
        self.total_execution_time += completion_time

        return ExecutionResult(
            status="success",
            location="edge",
            completion_time=completion_time
        )

    def get_utilization(self) -> float:
        """Get current edge utilization as a fraction (0.0 – 1.0)."""
        total_capacity = self.cpu_capacity * 100.0
        return self.current_cpu_usage / total_capacity